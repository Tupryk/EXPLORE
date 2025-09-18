import mujoco
import numpy as np
from explore.env.mujoco_sim import MjSim

sim = MjSim("configs/franka_emika_panda/scene.xml", 0.01, view=True, verbose=0)

joint_dim = sim.data.ctrl.shape
print(joint_dim)

tau_action = 1

ctrl_time = 0.0
q_target = np.array([
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255,
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255
])
sim.resetSplineRef(ctrl_time)
sim.setSplineRef(q_target.reshape(1, -1), [tau_action])
sim.step(tau_action, view=tau_action)
# print(sim.data.ctrl)
ctrl_time += tau_action

for i in range(1000):
    # print(sim.data.qpos[16:18])
    # print(sim.data.qpos.shape)
    sim.step_sim(True)
