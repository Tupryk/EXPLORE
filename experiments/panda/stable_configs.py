import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from explore.env.mujoco_sim import MjSim


h5_file = "configs/stable/grasp_configs.h5"
mujoco_xml = "configs/mujoco_/franka_emika_panda/pandas_table.xml"

# h5_file = "configs/stable/fingerBox.h5"
# mujoco_xml = "configs/mujoco_/fingerBox.xml"

# h5_file = "configs/stable/fingerRamp_onRamp.h5"
# mujoco_xml = "configs/mujoco_/fingerRamp.xml"

# h5_file = "configs/stable/new_rnd_twoFingers.h5"
# mujoco_xml = "configs/mujoco_/twoFingers.xml"

file = h5py.File(h5_file, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ball_x = stable_configs[:, -7]
# ball_y = stable_configs[:, -6]
# ball_z = stable_configs[:, -5]
# ax.scatter(ball_x, ball_y, ball_z, alpha=.2)
# plt.axis("equal")
# plt.show()

sim = MjSim(mujoco_xml, view=True, verbose=1, tau_sim=1e-3)

sampled_configs = np.random.randint(0, stable_configs.shape[0], (100))

# for i in sampled_configs:
for i, sc in enumerate(stable_configs):
    print(i)
    # sim.pushConfig(sc)
    # print(sc)
    # print(stable_configs_ctrl[i])
    q = stable_configs_ctrl[i].copy()
    # q[7] = 0
    # q[15] = 0
    sim.pushConfig(sc, q)
    time.sleep(1)
    sim.step(1, view=.5)
