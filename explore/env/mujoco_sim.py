# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import time, math
import numpy as np
import mujoco.viewer
from explore.utils.utils import ND_BSpline


class MjSim:

    def __init__(self, xml: str, tau_sim: float=0.01, view: bool=False, verbose: int=0):
        self.tau_sim = tau_sim
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.tau_sim
        self.verbose = verbose

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
            
        self.resetSplineRef(0.)

    def pushConfig(self, joint_state: np.ndarray, frame_poses: np.ndarray):
        qn = joint_state.shape[0]
        self.data.qpos[:qn] = joint_state
        self.data.qvel[:qn] = np.zeros(qn)

        for i, pose in enumerate(frame_poses):
            self.data.qpos[qn + 7 * i : qn + 7 * (i + 1)] = pose
            self.data.qvel[qn + 6 * i : qn + 6 * (i + 1)] = np.zeros(6)

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def run(self, steps: int, view: float=1.):
        
        for k in range(steps):
            # ref_pos = self.ctrl.eval(self.ctrl_time)
            perc = (k+1)/steps
            ref_pos = (1 - perc) * self.ctrl[0] + (perc) * self.ctrl[1]
            self.data.ctrl = ref_pos
            mujoco.mj_step(self.model, self.data)
            self.ctrl_time += self.tau_sim
            
            if view > 0.:
                viewSteps = math.ceil(0.03 / self.tau_sim / view)
                
                if ((k+1) % viewSteps==0 or k==steps-1):
                    if self.viewer is not None:
                        self.viewer.sync()
                    time.sleep(view * viewSteps * self.tau_sim)

    def step(self, tau: float, view: float):
        self.run(math.ceil(tau/self.tau_sim), view)

    def getState(self):
        state = (
            self.data.time,
            np.copy(self.data.qpos),
            np.copy(self.data.qvel),
            np.copy(self.data.act),
        )
        return state

    def setState(self, time: float, qpos: np.ndarray,
                 qvel: np.ndarray, act: np.ndarray):
        self.data.time = time
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.act = act
        mujoco.mj_forward(self.model, self.data)
        self.ctrl_time = time
        if self.viewer is not None:
            self.viewer.sync()

    def resetSplineRef(self, ctrl_time: float):
        ref = self.data.qpos[:self.data.ctrl.size]
        # self.ctrl = ND_BSpline(ctrl_time, ref, degree=1)
        self.ctrl = [ref]
        self.ctrl_dim = self.data.ctrl.size
        self.ctrl_time = ctrl_time

    def setSplineRef(self, points: np.ndarray,
                     times: np.ndarray, append: bool):
        if not append:
            # self.ctrl.compute(points, times, self.ctrl_time)
            self.ctrl.append(points[0])
        else:
            raise NotImplementedError("Appending not implemented")
        
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
