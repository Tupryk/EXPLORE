import os
import h5py
import numpy as np
from tqdm import trange
from omegaconf import DictConfig

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import explain_qpos


class GenStableConfigs:
    
    def __init__(self, cfg: DictConfig):
        
        self.output_dir = cfg.output_dir
        self.config_count = cfg.config_count
        self.verbose = cfg.verbose

        self.mujoco_xml = cfg.sim.mujoco_xml
        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        assert self.joints_are_same_as_ctrl
        
        self.sim = MjSim(
            self.mujoco_xml,
            self.tau_sim,
            view=False,
            verbose=0,
            interpolate=self.interpolate_actions,
            joints_are_same_as_ctrl=self.joints_are_same_as_ctrl,
            use_spline_ref=cfg.sim.use_spline_ref
        )
        if self.verbose:
            explain_qpos(self.sim.model)

        self.state_dim = self.sim.data.qpos.shape[0]
        self.ctrl_dim = self.sim.data.ctrl.shape[0]
        self.base_state = self.sim.data.qpos[:]
        self.vel_vec = np.zeros_like(self.sim.data.qvel)

    def run(self) -> tuple[list[np.ndarray], list[np.ndarray]]:

        cc = trange(self.config_count) if self.verbose else range(self.config_count)
        stable_configs = []
        stable_configs_ctrl = []

        for i in cc:

            attempt_count = 0
            
            while True:

                attempt_count += 1
                if self.verbose > 1:
                    print(f"Attempt number {attempt_count} for config {i}")
                
                qpos = self.base_state + 0.1 * np.random.randn(self.state_dim)
                ctrl = qpos[self.state_dim - self.ctrl_dim:]
                
                self.sim.setState(0.0, qpos, self.vel_vec, ctrl)
                self.sim.step(0.2)
                
                _, new_qpos, qvel, _, _, _ = self.sim.getState()
                
                e = new_qpos - qpos
                pos_cost = e.T @ e
                vel_cost = qvel.T @ qvel
                if pos_cost < 0.01 and vel_cost < 2:
                    stable_configs.append(qpos)
                    stable_configs_ctrl.append(ctrl)
                    break
                
                if self.verbose > 2:
                    print("Position cost: ", pos_cost)
                    print("Velocity cost: ", vel_cost)

        qpos_array = np.array(stable_configs)
        ctrl_array = np.array(stable_configs_ctrl)

        data_path = os.path.join(self.output_dir, "stable_configs.h5")
        with h5py.File(data_path, "w") as f:
            f.create_dataset("qpos", data=qpos_array)
            f.create_dataset("ctrl", data=ctrl_array)

        return stable_configs, stable_configs_ctrl
