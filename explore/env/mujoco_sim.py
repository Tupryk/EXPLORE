# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import time, math
import numpy as np
import mujoco.viewer

from explore.utils.mj import explain_qpos
from explore.utils.utils import ND_BSpline


class MjSim:

    def __init__(self, xml_path: str, tau_sim: float=1e-3, view: bool=False, interpolate: bool=False, verbose: int=0):
        
        self.tau_sim = tau_sim
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.tau_sim
        self.interpolate = interpolate
        self.verbose = verbose

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

    def pushConfig(self, joint_state: np.ndarray, ctrl_state: np.ndarray=None):
        self.data.qpos[:] = joint_state
        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        if not (ctrl_state is None):
            self.data.ctrl[:] = ctrl_state

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def step(self, tau_action: float,
             ctrl_target: np.ndarray=None, view: float=0.0):
        
        steps = math.ceil(tau_action/self.tau_sim)
        # This prev ctrl is not quite right, as you would have to take the qpos for a proper interpolation
        prev_ctrl = self.data.ctrl[:]

        if not self.interpolate and not (ctrl_target is None):
            self.data.ctrl[:] = ctrl_target
        
        for k in range(steps):
            
            if self.interpolate and not (ctrl_target is None):
                self.data.ctrl[:] = prev_ctrl * (1 - (k+1)/steps) + ctrl_target * ((k+1)/steps)
            
            mujoco.mj_step(self.model, self.data)
            
            if view > 0.:
                self.viewer.sync()
                time.sleep(view/steps)

    def getState(self):
        state = (
            self.data.time,
            np.copy(self.data.qpos),
            np.copy(self.data.qvel),
            np.copy(self.data.ctrl),
        )
        return state

    def setState(self, time: float, qpos: np.ndarray,
                 qvel: np.ndarray, ctrl: np.ndarray):
        self.data.time = time
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)
        self.ctrl_time = time
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
    
    def renderImg(self, w: int=640, h: int=480) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, w, h)
        renderer.update_scene(self.data)
        frame = renderer.render()
        return frame
