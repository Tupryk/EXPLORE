# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import time, math
import numpy as np
import mujoco.viewer

from explore.utils.mj import explain_qpos


class MjSim:

    def __init__(self,
                 xml_path: str,
                 tau_sim: float=1e-3,
                 interpolate: bool=False, 
                 joints_are_same_as_ctrl: bool=False,
                 view: bool=False,
                 use_spline_ref: bool=False,
                 verbose: int=0):
        
        self.tau_sim = tau_sim
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.tau_sim
        self.interpolate = interpolate
        self.joints_are_same_as_ctrl = joints_are_same_as_ctrl
        self.use_spline_ref = use_spline_ref
        self.verbose = verbose
        self.renderer = None
        self.ctrl_dim = self.data.ctrl.shape[0]

        self.frame_dt = 1.0 / 24.0
        self.next_frame_time = 0.0

        if view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
            
        if self.verbose:
            print(f"Loaded config '{xml_path}' with position values:")
            print(self.data.qpos)
            explain_qpos(self.model)
            
        frames = []

        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            frames.append(body_id)

        self.possible_contacts = []
        while len(frames):
            for i in range(1, len(frames)):
                self.possible_contacts.append((frames[0], frames[i]))
            del frames[0]
        
        self.spline_ref = None
        if self.use_spline_ref:
            self.spline_ref = ry.BSpline()
            self.spline_ref.set(2, self.data.ctrl.reshape(1, -1), [0.0])

    def pushConfig(self, joint_state: np.ndarray, ctrl_state: np.ndarray=None):
        self.data.time = 0.0
        self.next_frame_time = 0.0
        self.data.qpos[:] = joint_state
        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        
        if ctrl_state is None:
            assert self.joints_are_same_as_ctrl
            ctrl_state = joint_state[:self.ctrl_dim]
        
        self.data.ctrl[:] = ctrl_state
        if self.use_spline_ref:
            self.spline_ref.set(2, ctrl_state.reshape(1, -1), [0.0])

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()
        
    def step(self,
             tau_action: float,
             ctrl_target: np.ndarray=None,
             view: str="",
             log_all: bool=False) -> tuple[list, list, list]:
        
        if self.use_spline_ref and ctrl_target is not None:
            self.spline_ref.overwriteSmooth(
                ctrl_target.reshape(1, -1),
                np.array([2.*tau_action]),
                self.data.time
            )
            
        steps = math.ceil(tau_action/self.tau_sim)
        if self.joints_are_same_as_ctrl:
            prev_ctrl = self.data.qpos[:self.data.ctrl.shape[0]]
        else:
            # This prev ctrl is not quite right, as you would have to take the qpos for a proper interpolation
            prev_ctrl = self.data.ctrl[:]

        frames = []
        states = []
        ctrls = []
        if (not self.use_spline_ref) and not self.interpolate and not (ctrl_target is None):
            self.data.ctrl[:] = ctrl_target
        
        for k in range(steps):
            
            if ctrl_target is not None:
                
                if (not self.use_spline_ref) and self.interpolate:
                    perc = (k+1)/steps
                    self.data.ctrl[:] = prev_ctrl * (1 - perc) + ctrl_target * perc
                    
                elif self.use_spline_ref:
                    self.data.ctrl[:] = self.spline_ref.eval3(self.data.time)[0]
            
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
            np.copy(self.spline_ref)
        )
        return state

    def setState(self, time: float, qpos: np.ndarray,
                 qvel: np.ndarray, ctrl: np.ndarray, spline_ref=None):
        self.data.time = time
        self.next_frame_time = time
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        self.spline_ref = spline_ref
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
