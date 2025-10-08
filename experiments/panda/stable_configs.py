import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from explore.env.mujoco_sim import MjSim


file = h5py.File("configs/pandasTable_ball.h5", 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]

fig = plt.figure()
ax = plt.axes(projection='3d')
ball_x = stable_configs[:, -7]
ball_y = stable_configs[:, -6]
ball_z = stable_configs[:, -5]
ax.scatter(ball_x, ball_y, ball_z, alpha=.2)
plt.axis("equal")
plt.show()

sim = MjSim("configs/franka_emika_panda/scene.xml", view=True, verbose=1)

sampled_configs = np.random.randint(0, stable_configs.shape[0], (100))

# for i in sampled_configs:
for i, sc in enumerate(stable_configs):
    print(i)
    # sim.pushConfig(sc)
    sim.pushConfig(sc, stable_configs_ctrl[i])
    time.sleep(1)
    sim.step(1, view=.5)
