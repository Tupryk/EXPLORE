# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import time, math
import numpy as np
import mujoco.viewer
from omegaconf import DictConfig

from explore.utils.mj import explain_qpos, get_model_quaternions


class MjSim:

    def __init__(self, cfg: DictConfig, view: bool=False):
        
        self.verbose = cfg.get("verbose", 0)
        self.tau_sim = cfg.get("tau_sim", 1e-3)
        self.interpolate = cfg.get("interpolate", True)
        
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.tau_sim
        self.renderer = None
        self.ctrl_dim = self.data.ctrl.shape[0]
        
        if self.verbose:
            print(f"Loaded config '{cfg.xml_path}' with position values:")
            print(self.data.qpos)
            explain_qpos(self.model)

        self.q_mask = np.array(cfg.get("q_mask", []))
        self.dist_weight = cfg.get("dist_weight", 0.1)
        self.dist_max = cfg.get("dist_max", 0.2)
        self.vel_weight = cfg.get("velocity_weight", 0.0)

        obj_names = cfg.get("objs", [])
        self.objs = []
        for geom_name in obj_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.objs.append(geom_id)

        contact_names = cfg.get("contacts", [])
        self.contacts = []
        for geom_name in contact_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.contacts.append(geom_id)
        
        self.geoms_in_cost = []
        geoms_in_cost_names = cfg.get("geoms_in_cost", [])
        self.geoms_in_cost_weights = np.array(cfg.get("geoms_in_cost_weights", []))
        self.vels_geoms_in_cost_weights = np.array(cfg.get("vels_geoms_in_cost_weights", []))
        for geom_name in geoms_in_cost_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.geoms_in_cost.append(geom_id)
        
        self.custom_state_sequence = []
        self.custom_state_sequence_scaled = []
        if len(self.q_mask):
            def q_state():
                return np.copy(self.data.qpos)
            self.custom_state_sequence.append(q_state)
            self.custom_state_sequence_scaled.append(lambda: q_state() * self.q_mask)

        if self.vel_weight:
            def qvel_state() -> np.ndarray:
                return np.copy(self.data.qvel)
            self.custom_state_sequence.append(qvel_state)
            self.custom_state_sequence_scaled.append(lambda: qvel_state() * self.vel_weight)
        
        if self.geoms_in_cost:
            def geoms_state() -> np.ndarray:
                return self.data.geom_xpos[self.geoms_in_cost].ravel()
            self.custom_state_sequence.append(geoms_state)
            self.custom_state_sequence_scaled.append(lambda: geoms_state() * self.geoms_in_cost_weights)
        
        if len(self.vels_geoms_in_cost_weights):
            def geoms_vels_state() -> np.ndarray:
                vels = np.empty((len(self.geoms_in_cost), 3))

                for i, geom_id in enumerate(self.geoms_in_cost):
                    vel6 = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        self.model,
                        self.data,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                        vel6,
                        0,  # world frame
                    )
                    vels[i] = vel6[3:]

                vels = vels.ravel()
                return vels

            self.custom_state_sequence.append(geoms_vels_state)
            self.custom_state_sequence_scaled.append(lambda: geoms_vels_state() * self.vels_geoms_in_cost_weights)

        if self.objs:
            def dists_state() -> np.ndarray:
                fromto = np.zeros(6)
                dists = np.zeros(len(self.objs) * len(self.contacts))
                i = 0
                for oi in self.objs:
                    for ci in self.contacts:
                        dist = mujoco.mj_geomDistance(self.model, self.data, ci, oi, self.dist_max, fromto)
                        dists[i] = 1 - np.clip(dist / self.dist_max, 0., 1.)
                        i += 1
                return dists
            self.custom_state_sequence.append(dists_state)
            self.custom_state_sequence_scaled.append(lambda: dists_state() * self.dist_weight)

        if view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
            
        frames = []
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            frames.append(body_id)

        self.possible_contacts = []
        while len(frames):
            for i in range(1, len(frames)):
                self.possible_contacts.append((frames[0], frames[i]))
            del frames[0]
        
        self.model_quats = get_model_quaternions(self.model) if len(self.q_mask) else []
        
        self.frame_dt = 1.0 / 24.0
        self.next_frame_time = 0.0
        camera = cfg.get("camera", "")
        if camera:
            self.setupRenderer(cfg.render_w, cfg.render_h, camera=camera)
            
    def getCustomState(self) -> np.ndarray:
        return np.concatenate([cs() for cs in self.custom_state_sequence])
    
    def getCustomStateScaled(self) -> np.ndarray:
        return np.concatenate([cs() for cs in self.custom_state_sequence_scaled])
        
    def pushConfig(self, joint_state: np.ndarray, ctrl_state: np.ndarray=None):
        self.data.time = 0.0
        self.next_frame_time = 0.0
        self.data.qpos[:] = joint_state
        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        
        if ctrl_state is None:
            ctrl_state = joint_state[:self.ctrl_dim]
        
        self.data.ctrl[:] = ctrl_state

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()
        
    def step(self,
             tau_action: float,
             ctrl_target: np.ndarray=None,
             view: str="",
             log_all: bool=False) -> tuple[list, list, list]:
        
        steps = math.ceil(tau_action/self.tau_sim)
        prev_ctrl = np.copy(self.data.ctrl)

        frames = []
        states = []
        ctrls = []
        if not self.interpolate and ctrl_target is not None:
            self.data.ctrl[:] = ctrl_target
        
        for k in range(steps):
            
            if ctrl_target is not None:
                
                if self.interpolate:
                    perc = (k+1)/steps
                    self.data.ctrl[:] = prev_ctrl * (1 - perc) + ctrl_target * perc
            
            mujoco.mj_step(self.model, self.data)
            
            if view:
                if self.viewer != None:
                    self.viewer.sync()
                    time.sleep(tau_action/steps)

                else:
                    if self.data.time >= self.next_frame_time:
                        if not (self.renderer is None):
                            frames.append(self.renderImg(view))

                        if not log_all:
                            states.append(np.copy(self.data.qpos))

                        self.next_frame_time = self.next_frame_time + self.frame_dt
            if log_all:
                states.append(np.copy(self.data.qpos))
                ctrls.append(np.copy(self.data.ctrl))
        
        return frames, states, ctrls
        
    def getState(self):

        state = (
            self.data.time,
            np.copy(self.data.qpos),
            np.copy(self.data.qvel),
            np.copy(self.data.ctrl),
        )
        return state

    def setState(self,
                 time: float,
                 qpos: np.ndarray,
                 qvel: np.ndarray,
                 ctrl: np.ndarray):
        
        self.data.time = time
        self.next_frame_time = time
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)  # Investigate!!
        if self.viewer is not None:
            self.viewer.sync()
        
    def getContacts(self) -> np.ndarray:
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            contacts.append((body1, body2))
        
        contacts_vec = np.zeros(len(self.possible_contacts))
        for c in contacts:
            if c in self.possible_contacts:
                contacts_vec[self.possible_contacts.index(c)] = 1
            else:
                contacts_vec[self.possible_contacts.index((c[1], c[0]))] = 1
        
        return contacts_vec
    
    def setupRenderer(self, w: int=640, h: int=480, camera: str=""):
        self.renderer = mujoco.Renderer(self.model, h, w)
        self.camera = camera

    def renderImg(self, other_camera: str="") -> np.ndarray:
        # TODO: Make this nicer
        if self.camera:
            if other_camera:
                self.renderer.update_scene(self.data, other_camera)
            else:
                self.renderer.update_scene(self.data, self.camera)
        else:
            self.renderer.update_scene(self.data)
        frame = self.renderer.render()
        return frame
