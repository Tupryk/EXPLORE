import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from explore.env.mujoco_sim import MjSim


# h5_file = "configs/pandasTable_ball.h5"
# mujoco_xml = "configs/franka_emika_panda/scene.xml"
h5_file = "configs/fingerBox.h5"
mujoco_xml = "configs/fingerBox.xml"

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

sim = MjSim(mujoco_xml, view=True, verbose=1)

sampled_configs = np.random.randint(0, stable_configs.shape[0], (100))

# for i in sampled_configs:
for i, sc in enumerate(stable_configs):
    print(i)
    # sim.pushConfig(sc)
    print(sc)
    print(stable_configs_ctrl[i])
    sim.pushConfig(sc, stable_configs_ctrl[i])
    time.sleep(1)
    sim.step(1, view=.5)
