import time
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from explore.env.mujoco_sim import MjSim

sim = MjSim("configs/franka_emika_panda/panda_single.xml", 0.001, view=True, verbose=0)

ctrl_joint_ids = [
    sim.model.jnt_qposadr[sim.model.actuator_trnid[i][0]]
    for i in range(sim.model.nu)
]
ctrl_joint_ids = [int(sim.model.actuator_trnid[i][0]) for i in range(sim.model.nu)]
print(ctrl_joint_ids)
motor_qpos = [float(sim.data.qpos[sim.model.jnt_qposadr[jid]]) for jid in ctrl_joint_ids]
print(motor_qpos)

joint_dim = sim.data.ctrl.shape
print(joint_dim)

tau_action = 10

ctrl_time = 0.0
q_target = np.array([
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255,
    # 0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255
])
config = np.array([
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
    0, 0, 0.7, 1, 0, 0, 0
])
# sim.pushConfig(config)
time.sleep(5)
sim.step(tau_action, ctrl_target=q_target, view=tau_action)
sim.setupRenderer(camera="l_wrist_cam")
img = sim.renderImg()
plt.imsave("experiments/wrist_img.png", img)
