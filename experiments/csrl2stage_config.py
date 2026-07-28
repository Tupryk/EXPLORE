import h5py
import numpy as np

h5_file_in = "data/double_sphere.h5"
h5_file_out = "configs/stable/double_sphere_csrl.h5"

file = h5py.File(h5_file_in, 'r')
stable_configs = file["q"]
print(stable_configs.shape)

stable_configs = np.array([[*sc, 1., 0., 0., 0.] for sc in stable_configs])
print(stable_configs.shape)

with h5py.File(h5_file_out, "w") as f:
    f.create_dataset("qpos", data=stable_configs)
    f.create_dataset("ctrl", data=stable_configs[:, :3])
    