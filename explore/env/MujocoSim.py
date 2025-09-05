# https://mujoco.readthedocs.io/en/stable/python.html

import mujoco
import mujoco.viewer
import numpy as np
import time, math
import robotic as ry


class MjSim:
    tau_sim = 0.01

    def __init__(self, xml, C: ry.Config, view=True, verbose: int=1):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.tau_sim

        if view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.freeobjs = []
        for f in C.getRoots():
            if 'mass' in f.asDict():
                self.freeobjs.append(f)
        if verbose:
            print(f'-- initializing MjSim with {len(self.freeobjs)} free objects and joint dimension {C.getJointDimension()}')
        self.C = C
        self.pushConfig()

        assert self.data.qpos.size == self.C.getJointDimension() + 7 * len(self.freeobjs)
        assert self.data.ctrl.size == self.C.getJointDimension()

        self.resetSplineRef(0.)

    def pushConfig(self):
        qn = self.C.getJointDimension()
        self.data.qpos[:qn] = self.C.getJointState()
        self.data.qvel[:qn] = np.zeros(qn)

        for i, f in enumerate(self.freeobjs):
            self.data.qpos[qn + 7 * i : qn + 7 * (i + 1)] = f.getPose()
            self.data.qvel[qn + 6 * i : qn + 6 * (i + 1)] = np.zeros(6)

        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        # mujoco.mj_setState(self.m,self.d,q, mujoco.mjtState.mjSTATE_QPOS)
        # mujoco.mj_setState(m,d,q, mujoco.mjtState.mjSTATE_CTRL)

    def pullConfig(self):
        qpos = self.data.qpos
        qn = self.C.getJointDimension()

        self.C.setJointState(qpos[:qn])

        start = qn
        for f in self.freeobjs:
            f.setPose(qpos[start:start + 7])
            start += 7

    def run(self, steps, view=1.):
        for k in range(steps):
            ref = self.ctrl.eval3(self.ctrl_time)
            self.data.ctrl = ref[0]
            mujoco.mj_step(self.model, self.data)
            self.ctrl_time += self.tau_sim
            if view>0.:
                viewSteps = math.ceil(0.03 / self.tau_sim / view)
                if ((k+1)%viewSteps==0 or k==steps-1):
                    if self.viewer is not None:
                        self.viewer.sync()
                    self.pullConfig()
                    self.C.view(False, f"mujoco sim time: {self.data.time:6.3f}, ctrl time: {self.ctrl_time:6.3f}")
                    time.sleep(view * viewSteps * self.tau_sim)

    def step(self, u, tau, mode, view):
        self.run(math.ceil(tau/self.tau_sim), view)
        self.pullConfig()

    def getJustFrameState(self):
        return self.C.getFrameState()

    def getState(self):
        state = (
            self.C.getFrameState(),
            np.copy(self.data.qpos),
            self.data.time,
            np.copy(self.data.qvel),
            np.copy(self.data.act),
        )
        return state

    def setState(self, frame_state, qpos, time, qvel, act):
        self.data.time = time
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.act = act
        mujoco.mj_forward(self.model, self.data)
        self.ctrl_time = time
        if self.viewer is not None:
            self.viewer.sync()
        self.pullConfig()

    def resetSplineRef(self, ctrl_time):
        self.ctrl = ry.BSpline()
        ref = self.data.qpos
        ref = ref[:self.data.ctrl.size]
        self.ctrl.set(2, ref.reshape(1, -1), [ctrl_time])
        self.ctrl_dim = self.data.ctrl.size
        self.ctrl_time = ctrl_time

    def setSplineRef(self, points, times, append):
        if not append:
            self.ctrl.overwriteSmooth(points, times, self.ctrl_time)
        else:
            raise NotImplementedError()
