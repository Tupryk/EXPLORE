import time
import mujoco
import numpy as np


tau_sim = 0.001
model = mujoco.MjModel.from_xml_path("configs/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)
model.opt.timestep = tau_sim
print(data.qpos)
data.qpos = q_target = np.array([
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
    0, 0, 0.7, 1, 0, 0, 0
])
data.ctrl = np.array([
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255,
    0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255
])
print(data.qpos)
viewer = mujoco.viewer.launch_passive(model, data)
print(model.nu)
print(model.actuator_trnid)
ctrl_joint_ids = [int(model.actuator_trnid[i][0]) for i in range(model.nu)]
print(ctrl_joint_ids)
motor_qpos = [float(data.qpos[model.jnt_qposadr[jid]]) for jid in ctrl_joint_ids]
print(motor_qpos)
ctrl_ranges = model.actuator_ctrlrange
print("ctrl_ranges: ", ctrl_ranges)

for i in range(10000):
    if i > 3000:
        mujoco.mj_step(model, data)
        viewer.sync()
    time.sleep(tau_sim)
