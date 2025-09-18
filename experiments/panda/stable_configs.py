import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

from explore.env.mujoco_sim import MjSim


file = h5py.File("configs/pandasTable_ball.h5", 'r')
stable_configs = file["positions"]

fig = plt.figure()
ax = plt.axes(projection='3d')
ball_x = stable_configs[:, -7]
ball_y = stable_configs[:, -6]
ball_z = stable_configs[:, -5]
ax.scatter(ball_x, ball_y, ball_z, alpha=.2)
plt.axis("equal")
plt.show()

sim = MjSim("configs/franka_emika_panda/scene.xml", 0.01, view=True, verbose=1)

sampled_configs = np.random.randint(0, stable_configs.shape[0], (10))

for i in sampled_configs:
    print(stable_configs[i])
    sim.pushConfig(stable_configs[i])
    time.sleep(4)
