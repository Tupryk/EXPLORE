import time
import h5py
import numpy as np
# import robotic as ry

from explore.env.mujoco_sim import MjSim


old_file_path = "configs/rnd_twoFingers.h5"
new_file_path = "configs/new_rnd_twoFingers.h5"
file = h5py.File(old_file_path, 'r')

# C = ry.Config()
# C.addFile("configs/twoFingers.g")
# for p in file["positions"]:
#     C.setJointState(p[3:])
#     C.getFrame("obj").setPosition(p[:3])
#     C.view(True)

data_pos = []
for p in file["positions"]:
    pos = np.concatenate((p[:3], np.array([1., 0., 0., 0.])))
    pos = np.concatenate((p[3:], pos))
    data_pos.append(pos)
data_pos = np.array(data_pos)
data_ctrl = file["positions"][:, 3:]

with h5py.File(new_file_path, "w") as f:
    f.create_dataset("qpos", data=data_pos)
    f.create_dataset("ctrl", data=data_ctrl)

file = h5py.File(new_file_path, 'r')
stable_configs = file["qpos"]
stable_configs_ctrl = file["ctrl"]

sim = MjSim("configs/twoFingers.xml", view=True, verbose=1)

sampled_configs = np.random.randint(0, stable_configs.shape[0], (100))

# for i in sampled_configs:
for i, sc in enumerate(stable_configs):
    print(i)
    # sim.pushConfig(sc)
    sim.pushConfig(sc, stable_configs_ctrl[i])
    time.sleep(1)
    sim.step(1, view=.5)