import h5py
import numpy as np

from explore.utils.vis import AdjMap

# txt_file = "data/joint_states.txt"
# h5_file = "configs/pandasTable_ball.h5"
# txt_file = "data/joint_states_fingerBox.txt"
# h5_file = "configs/fingerBox.h5"
txt_file = "data/grasp_configs.txt"
h5_file = "configs/grasp_configs.h5"

# SAME_THRESH = 0.07
SAME_THRESH = 0.05

data = np.loadtxt(txt_file, dtype=np.float64)

new_data_pos = []
new_data_ctrl = []
for i, vec in enumerate(data):
    state_vec = np.zeros(25)
    
    state_vec[:8] = vec[:8]
    state_vec[8] = vec[7]
    state_vec[9:17] = vec[8:16]
    state_vec[17] = vec[15]
    
    state_vec[18:21] = vec[16:]
    state_vec[21] = 1
    
    ctrl_vec = np.zeros(16)
    ctrl_vec[:8] = state_vec[:8]
    ctrl_vec[8:16] = state_vec[9:17]
    ctrl_vec[7] *= 255/0.04
    ctrl_vec[15] *= 255/0.04

    # state_vec = vec
    # ctrl_vec = vec[:3]
    
    new_data_pos.append(state_vec)
    new_data_ctrl.append(ctrl_vec)

data_pos = np.array(new_data_pos)
data_ctrl = np.array(new_data_ctrl)

i = 0
while i < data_pos.shape[0]:
    keep = []
    for j in range(data_pos.shape[0]):
        if i == j: keep.append(i)
        
        e = data_pos[i] - data_pos[j]
        if e.T @ e > SAME_THRESH:
            keep.append(j)

    data_pos = data_pos[keep]
    data_ctrl = data_ctrl[keep]
    i += 1

costs = np.zeros((data_pos.shape[0], data_pos.shape[0]))
for i in range(data_pos.shape[0]):
    for j in range(data_pos.shape[0]):
        
        if i == j: continue
        
        e = data_pos[i] - data_pos[j]
        costs[i, j] = e.T @ e

masked = costs.copy()
np.fill_diagonal(masked, np.inf)
print(f"Config Count: {data_pos.shape[0]}")
print("Min cost: ", masked.min())
print("Max cost: ", costs.max())

with h5py.File(h5_file, "w") as f:
    f.create_dataset("qpos", data=data_pos)
    f.create_dataset("ctrl", data=data_ctrl)

print(f"Success: Data saved to {h5_file}.")

AdjMap(costs, SAME_THRESH, costs.max())
