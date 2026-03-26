import os
import math
import h5py
import mujoco
import warp as wp
import numpy as np
from tqdm import trange
import mujoco_warp as mjw
from omegaconf import DictConfig

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import explain_qpos


class GenStableConfigs:
    
    def __init__(self, cfg: DictConfig):
        
        self.output_dir = cfg.output_dir
        self.config_count = cfg.config_count
        self.verbose = cfg.verbose
        self.use_gpu = cfg.use_gpu

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

        if self.use_gpu:
            self.sim_instances = cfg.sim_instances
            self.warp_model = mjw.put_model(self.sim.model)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            # self.warp_data = mjw.put_data(self.sim.model, self.sim.data, nworld=self.sim_instances, njmax=500, nconmax=500)
            self.warp_data = mjw.put_data(self.sim.model, self.sim.data, nworld=self.sim_instances)
        
        if self.verbose:
            explain_qpos(self.sim.model)

        self.state_dim = self.sim.data.qpos.shape[0]
        self.ctrl_dim = self.sim.data.ctrl.shape[0]
        self.base_state = self.sim.data.qpos[:]

        self.sigma = 0.1

    def run(self) -> tuple[list[np.ndarray], list[np.ndarray]]:

        stable_configs = []
        stable_configs_ctrl = []
        zero_qvel = np.zeros_like(self.sim.data.qvel)

        if self.use_gpu:
            zero_qvel = wp.array(zero_qvel, dtype=wp.float32)
            found_count = 0
            while True:
                qpos = self.sigma * np.random.randn(self.state_dim * self.sim_instances)
                qpos = qpos.reshape(self.sim_instances, self.state_dim)
                qpos += self.base_state
                ctrl = qpos[:, self.state_dim - self.ctrl_dim:]

                self.warp_data.time.fill_(0.0)
                wp.copy(self.warp_data.qpos, wp.array(qpos, dtype=wp.float32))
                wp.copy(self.warp_data.qvel, zero_qvel)
                wp.copy(self.warp_data.ctrl, wp.array(ctrl, dtype=wp.float32))

                mjw.reset_data(self.warp_model, self.warp_data)

                steps = math.ceil(self.tau_action / self.tau_sim)

                for _ in range(steps):
                    mjw.step(self.warp_model, self.warp_data)
                
                new_qpos = self.warp_data.qpos.numpy()
                qvel = self.warp_data.qvel.numpy()
                
                es = new_qpos - qpos
                for i in range(self.sim_instances):
                    e = es[i]
                    pos_cost = e.T @ e
                    vel_cost = qvel[i].T @ qvel[i]

                    if pos_cost < 0.01 and vel_cost < 2:
                        stable_configs.append(np.copy(qpos[i]))
                        stable_configs_ctrl.append(np.copy(ctrl[i]))
                        
                        found_count += 1
                        if found_count >= self.config_count:
                            break
                        
                    if self.verbose > 2:
                        print("Position cost: ", pos_cost)
                        print("Velocity cost: ", vel_cost)

                if found_count >= self.config_count:
                    break

                print(f"Found {found_count} configs!")
            
        else:
            cc = trange(self.config_count) if self.verbose else range(self.config_count)
            for i in cc:

                attempt_count = 0
                
                while True:

                    attempt_count += 1
                    if self.verbose > 1:
                        print(f"Attempt number {attempt_count} for config {i}")
                    
                    qpos = self.base_state + self.sigma * np.random.randn(self.state_dim)
                    ctrl = qpos[self.state_dim - self.ctrl_dim:]
                    
                    self.sim.setState(0.0, qpos, zero_qvel, ctrl)
                    self.sim.step(self.tau_action)
                    
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
