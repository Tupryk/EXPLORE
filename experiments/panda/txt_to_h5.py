import h5py
import numpy as np

from explore.utils.vis import AdjMap

# txt_file = "data/joint_states.txt"
# h5_file = "configs/pandasTable_ball.h5"
# txt_file = "data/joint_states_fingerBox.txt"
# h5_file = "configs/fingerBox.h5"
# txt_file = "data/grasp_configs.txt"
# h5_file = "configs/grasp_configs.h5"
txt_file = "data/joint_states_unitree.txt"
h5_file = "configs/stable/g1.h5"

# SAME_THRESH = 0.07
SAME_THRESH = 0.05

data = np.loadtxt(txt_file, dtype=np.float64)

new_data_pos = []
new_data_ctrl = []
for i, vec in enumerate(data):
    if i >= 200: break
    # ### FRANKAS BOX ###
    # state_vec = np.zeros(25)
    
    # state_vec[:8] = vec[:8]
    # state_vec[8] = vec[7]
    # state_vec[9:17] = vec[8:16]
    # state_vec[17] = vec[15]
    
    # state_vec[18:21] = vec[16:]
    # state_vec[21] = 1
    
    # ctrl_vec = np.zeros(16)
    # ctrl_vec[:8] = state_vec[:8]
    # ctrl_vec[8:16] = state_vec[9:17]
    # ctrl_vec[7] *= 255/0.04
    # ctrl_vec[15] *= 255/0.04

    # state_vec = vec
    # ctrl_vec = vec[:3]
    
    ### UNITREE ###
    state_vec = np.zeros(36)
    ctrl_vec = np.zeros(29)
    
    state_vec[:7] = vec[:7]
    state_vec[2] += 0.69
    
    # 19 -  0 waist_yaw_joint
    state_vec[19] = vec[7]
    #  7 -  1 left_hip_pitch_joint
    state_vec[7] = vec[8]
    # 13 -  2 right_hip_pitch_joint
    state_vec[13] = vec[9]
    # 20 -  3 waist_roll_joint
    state_vec[20] = vec[10]
    #  8 -  4 left_hip_roll_joint
    state_vec[8] = vec[11]
    # 14 -  5 right_hip_roll_joint
    state_vec[14] = vec[12]
    # 21 -  6 waist_pitch_joint
    state_vec[21] = vec[13]
    #  9 -  7 left_hip_yaw_joint
    state_vec[9] = vec[14]
    # 15 -  8 right_hip_yaw_joint
    state_vec[15] = vec[15]
    # 22 -  9 left_shoulder_pitch_joint
    state_vec[22] = vec[16]
    # 29 -  0 right_shoulder_pitch_joint
    state_vec[29] = vec[17]
    # 10 - 11 left_knee_joint
    state_vec[10] = vec[18]
    # 16 - 12 right_knee_joint
    state_vec[16] = vec[19]
    # 23 - 13 left_shoulder_roll_joint
    state_vec[23] = vec[20]
    # 30 - 14 right_shoulder_roll_joint
    state_vec[30] = vec[21]
    # 11 - 15 left_ankle_pitch_joint
    state_vec[11] = vec[22]
    # 17 - 16 right_ankle_pitch_joint
    state_vec[17] = vec[23]
    # 24 - 17 left_shoulder_yaw_joint
    state_vec[24] = vec[24]
    # 31 - 18 right_shoulder_yaw_joint
    state_vec[31] = vec[25]
    # 12 - 19 left_ankle_roll_joint
    state_vec[12] = vec[26]
    # 18 - 20 right_ankle_roll_joint
    state_vec[18] = vec[27]
    # 25 - 21 left_elbow_joint
    state_vec[25] = vec[28]
    # 32 - 22 right_elbow_joint
    state_vec[32] = vec[29]
    # 26 - 23 left_wrist_roll_joint
    state_vec[26] = vec[30]
    # 33 - 24 right_wrist_roll_joint
    state_vec[33] = vec[31]
    # 27 - 25 left_wrist_pitch_joint
    state_vec[27] = vec[32]
    # 34 - 26 right_wrist_pitch_joint
    state_vec[34] = vec[33]
    # 28 - 27 left_wrist_yaw_joint
    state_vec[28] = vec[34]
    # 35 - 28 right_wrist_yaw_joint
    state_vec[35] = vec[35]
    
    ctrl_vec[12] = vec[7]
    ctrl_vec[0] = vec[8]
    ctrl_vec[6] = vec[9]
    ctrl_vec[13] = vec[10]
    ctrl_vec[1] = vec[11]
    ctrl_vec[7] = vec[12]
    ctrl_vec[14] = vec[13]
    ctrl_vec[2] = vec[14]
    ctrl_vec[8] = vec[15]
    ctrl_vec[15] = vec[16]
    ctrl_vec[22] = vec[17]
    ctrl_vec[3] = vec[18]
    ctrl_vec[9] = vec[19]
    ctrl_vec[16] = vec[20]
    ctrl_vec[23] = vec[21]
    ctrl_vec[4] = vec[22]
    ctrl_vec[10] = vec[23]
    ctrl_vec[17] = vec[24]
    ctrl_vec[24] = vec[25]
    ctrl_vec[5] = vec[26]
    ctrl_vec[11] = vec[27]
    ctrl_vec[18] = vec[28]
    ctrl_vec[25] = vec[29]
    ctrl_vec[19] = vec[30]
    ctrl_vec[26] = vec[31]
    ctrl_vec[20] = vec[32]
    ctrl_vec[27] = vec[33]
    ctrl_vec[21] = vec[34]
    ctrl_vec[28] = vec[35]
    
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
