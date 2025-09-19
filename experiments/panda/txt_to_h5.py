import h5py
import numpy as np

from explore.utils.vis import AdjMap

txt_file = "data/joint_states.txt"
h5_file = "configs/pandasTable_ball.h5"


data = np.loadtxt(txt_file, dtype=np.float64)

new_data = []
for i, vec in enumerate(data):
    new_vec = np.zeros(25)
    
    new_vec[:8] = vec[:8]
    new_vec[8] = vec[7]
    new_vec[9:17] = vec[8:16]
    new_vec[17] = vec[15]
    
    new_vec[18:21] = vec[16:]
    new_vec[21] = 1
    
    new_data.append(new_vec)

data = np.array(new_data)

SAME_THRESH = 0.1
i = 0
while i < data.shape[0]:
    keep = []
    for j in range(data.shape[0]):
        if i == j: keep.append(i)
        
        e = data[i] - data[j]
        if e.T @ e > SAME_THRESH:
            keep.append(j)

    data = data[keep]
    i += 1

costs = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        
        if i == j: continue
        
        e = data[i] - data[j]
        costs[i, j] = e.T @ e

masked = costs.copy()
np.fill_diagonal(masked, np.inf)
print(f"Data size: {data.shape[0]}")
print("Min cost: ", masked.min())
print("Max cost: ", costs.max())

with h5py.File(h5_file, "w") as f:
    f.create_dataset("positions", data=data)

print(f"Data saved to {h5_file} under key 'positions'.")

AdjMap(costs, SAME_THRESH, costs.max())

