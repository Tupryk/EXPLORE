import math
import h5py
import torch
import mujoco
import warp as wp
import numpy as np
from tqdm import tqdm
import mujoco_warp as mjw


class OptimPair:
    def __init__(self):
        NWORLD = 256

        new_file_path = "configs/stable/humanoid_box_grasps.h5"
        mujoco_xml = "configs/mujoco_/unitree_g1/table_box_scene.xml"

        self.start_idx = 1
        end_idx = 12217
        self.ctrl_n = 10
        tau_action = 0.1
        tau_sim = 0.005
        self.action_steps = math.ceil(tau_action / tau_sim)

        mj_model = mujoco.MjModel.from_xml_path(mujoco_xml)
        mj_model.opt.timestep = tau_sim
        mj_model.opt.ccd_iterations = 200
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, mj_data)

        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mj_data, nworld=NWORLD, nconmax=250, njmax=250)

        ctrl_torch = torch.zeros((NWORLD, mj_model.nu), device="cuda")
        ctrl_wp = wp.from_torch(ctrl_torch)

        file = h5py.File(new_file_path, 'r')
        stable_configs = file["qpos"]
        stable_configs_ctrl = file["ctrl"]

        data.qpos = wp.array(np.tile(stable_configs[self.start_idx], (NWORLD, 1)), dtype=wp.float32)
        data.ctrl = wp.array(np.tile(stable_configs_ctrl[self.start_idx], (NWORLD, 1)), dtype=wp.float32)
        mjw.forward(model, data)

    def run(self):
        ctrl_sequence = [torch.tensor(stable_configs_ctrl[self.start_idx], device="cuda")]
        for _ in range(self.ctrl_n-1):
            ctrl_target = torch.rand((NWORLD, mj_model.nu), device="cuda") * 2 - 1
            ctrl_sequence.append(ctrl_target)

        for target_ctrl_idx in tqdm(range(1, len(ctrl_sequence))):
            
            for t in range(self.action_steps):

                alpha = t / (self.action_steps - 1)

                ctrl_torch[:] = (1 - alpha) * ctrl_sequence[target_ctrl_idx-1] + alpha * ctrl_sequence[target_ctrl_idx]

                data.ctrl.assign(ctrl_wp)
                mjw.step(model, data)
