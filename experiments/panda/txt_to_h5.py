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
    # Joint 1: l_left_hip_pitch_joint, type=3, qpos address=7, dof=hinge(1)
    # Joint 2: l_left_hip_roll_joint, type=3, qpos address=8, dof=hinge(1)
    # Joint 3: l_left_hip_yaw_joint, type=3, qpos address=9, dof=hinge(1)
    
    # Joint 4: l_left_knee_joint, type=3, qpos address=10, dof=hinge(1)
    
    # Joint 5: l_left_ankle_pitch_joint, type=3, qpos address=11, dof=hinge(1)
    # Joint 6: l_left_ankle_roll_joint, type=3, qpos address=12, dof=hinge(1)
    
    # Joint 7: l_right_hip_pitch_joint, type=3, qpos address=13, dof=hinge(1)
    # Joint 8: l_right_hip_roll_joint, type=3, qpos address=14, dof=hinge(1)
    # Joint 9: l_right_hip_yaw_joint, type=3, qpos address=15, dof=hinge(1)
    
    # Joint 10: l_right_knee_joint, type=3, qpos address=16, dof=hinge(1)
    
    # Joint 11: l_right_ankle_pitch_joint, type=3, qpos address=17, dof=hinge(1)
    # Joint 12: l_right_ankle_roll_joint, type=3, qpos address=18, dof=hinge(1)
    
    # Joint 13: l_waist_yaw_joint, type=3, qpos address=19, dof=hinge(1)
    # Joint 14: l_waist_roll_joint, type=3, qpos address=20, dof=hinge(1)
    # Joint 15: l_waist_pitch_joint, type=3, qpos address=21, dof=hinge(1)
    
    # Joint 16: l_left_shoulder_pitch_joint, type=3, qpos address=22, dof=hinge(1)
    # Joint 17: l_left_shoulder_roll_joint, type=3, qpos address=23, dof=hinge(1)
    # Joint 18: l_left_shoulder_yaw_joint, type=3, qpos address=24, dof=hinge(1)
    
    # Joint 19: l_left_elbow_joint, type=3, qpos address=25, dof=hinge(1)
    
    # Joint 20: l_left_wrist_roll_joint, type=3, qpos address=26, dof=hinge(1)
    # Joint 21: l_left_wrist_pitch_joint, type=3, qpos address=27, dof=hinge(1)
    # Joint 22: l_left_wrist_yaw_joint, type=3, qpos address=28, dof=hinge(1)
    
    # Joint 23: l_right_shoulder_pitch_joint, type=3, qpos address=29, dof=hinge(1)
    # Joint 24: l_right_shoulder_roll_joint, type=3, qpos address=30, dof=hinge(1)
    # Joint 25: l_right_shoulder_yaw_joint, type=3, qpos address=31, dof=hinge(1)
    
    # Joint 26: l_right_elbow_joint, type=3, qpos address=32, dof=hinge(1)
    
    # Joint 27: l_right_wrist_roll_joint, type=3, qpos address=33, dof=hinge(1)
    # Joint 28: l_right_wrist_pitch_joint, type=3, qpos address=34, dof=hinge(1)
    # Joint 29: l_right_wrist_yaw_joint, type=3, qpos address=35, dof=hinge(1)
    state_vec[:7] = vec[:7]
    # Waist
    state_vec[19] = vec[7]
    state_vec[7] = vec[8]
    state_vec[13] = vec[9]
    state_vec[20] = vec[10]
    state_vec[8] = vec[11]
    state_vec[14] = vec[12]
    state_vec[21] = vec[13]
    state_vec[9] = vec[14]
    state_vec[15] = vec[15]
    state_vec[22] = vec[16]
    state_vec[29] = vec[17]
    state_vec[10] = vec[18]
    state_vec[16] = vec[19]
    state_vec[23] = vec[20]
    state_vec[30] = vec[21]
    state_vec[11] = vec[22]
    state_vec[17] = vec[23]
    state_vec[24] = vec[24]
    state_vec[31] = vec[25]
    state_vec[12] = vec[26]
    state_vec[18] = vec[27]
    state_vec[25] = vec[28]
    state_vec[32] = vec[29]
    state_vec[26] = vec[30]
    state_vec[33] = vec[31]
    state_vec[27] = vec[32]
    state_vec[34] = vec[33]
    state_vec[28] = vec[34]
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
