import h5py
import numpy as np

h5_file_small = "configs/stable/humanoid_box_grasps.h5"
h5_file_big = "configs/stable/humanoid_box_grasps_big.h5"

file = h5py.File(h5_file_small, 'r')
qpos = file["qpos"]
ctrl = file["ctrl"]

max_qpos_x = qpos[:, 0].max()
max_qpos_y = qpos[:, 1].max()
min_qpos_x = qpos[:, 0].min()
min_qpos_y = qpos[:, 1].min()

min_qpos_x = -0.5
max_qpos_x = 0.5
min_qpos_y = -0.5
max_qpos_y = 0.5

print(f"x-Range: [{min_qpos_x}, {max_qpos_x}]")
print(f"y-Range: [{min_qpos_y}, {max_qpos_y}]")

expanded_qpos = []
expanded_ctrl = []
small_i = 0
for i in range(100_000):
    small_i = i % len(qpos)
    
    qpos_mod = qpos[small_i].copy()
    qpos_mod[0] = np.random.uniform(min_qpos_x, max_qpos_x)
    qpos_mod[1] = np.random.uniform(min_qpos_y, max_qpos_y)
    
    qpos_mod[-7] += qpos_mod[0] - qpos[small_i, 0]
    qpos_mod[-6] += qpos_mod[1] - qpos[small_i, 1]
    
    expanded_qpos.append(qpos_mod)
    expanded_ctrl.append(ctrl[small_i].copy())
    
with h5py.File(h5_file_big, "w") as f:
    f.create_dataset("qpos", data=np.array(expanded_qpos))
    f.create_dataset("ctrl", data=np.array(expanded_ctrl))
