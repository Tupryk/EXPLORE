import h5py
import numpy as np

txt_file = "data/joint_states.txt"
h5_file = "configs/pandasTable_ball.h5"

data = np.loadtxt(txt_file, dtype=np.float64)

new_data = []
for i, vec in enumerate(data):
    if i >= 100: break
    new_vec = np.zeros(25)
    new_vec[:7] = vec[:7]
    new_vec[9:16] = vec[7:14]
    new_vec[18:21] = vec[14:17]
    new_vec[21] = 1
    new_data.append(new_vec)

data = np.array(new_data)

with h5py.File(h5_file, "w") as f:
    f.create_dataset("positions", data=data)

print(f"Data saved to {h5_file} under key 'positions'.")
