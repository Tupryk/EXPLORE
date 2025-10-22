import os
import h5py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from explore.env.mujoco_sim import MjSim
from explore.utils.utils import randint_excluding, extract_ball_from_img
from explore.datasets.utils import load_trees, generate_adj_map, get_feasible_paths


class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()

        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        self.mujoco_xml = cfg.sim.mujoco_xml
        
        self.stepsize = cfg.stepsize
        self.max_steps = cfg.max_steps  # Gets overwriten if use_guiding = True
        
        self.actions_noise_sigma = cfg.actions_noise_sigma
        self.use_vel = cfg.use_vel
        self.guiding = cfg.guiding
        self.reward = None
        
        self.use_vision = cfg.use_vision
        assert (self.use_vision and not self.use_vel) or (not self.use_vision and self.use_vel)

        # Setup sim
        self.start_config_idx = cfg.start_config_idx
        self.end_config_idx = cfg.target_config_idx
        self.goal_conditioning = cfg.goal_conditioning
        if not self.goal_conditioning and cfg.target_config_idx == -1:
            raise Exception("Setting many goals but no goal conditioning!")
        if self.goal_conditioning and cfg.target_config_idx != -1:
            raise Exception("Using goal conditioning but only a single goal!")
        
        self.stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.stable_configs["qpos"].shape[0]
        
        self.verbose = cfg.verbose

        config_path = os.path.join(cfg.trajectory_data_path, ".hydra/config.yaml")
        trees_cfg = OmegaConf.load(config_path)
        self.q_mask = np.array(trees_cfg.RRT.q_mask)

        self.guiding_path = []
        if self.guiding:
            
            tree_dataset = os.path.join(cfg.trajectory_data_path, "trees")
            self.trees, _, _ = load_trees(tree_dataset)
            
            min_costs, top_nodes = generate_adj_map(self.trees, self.q_mask)
            self.traj_pairs, self.traj_end_nodes, _ = get_feasible_paths(
                min_costs, top_nodes, self.start_config_idx, self.end_config_idx, trees_cfg.RRT.min_cost)
            
            if not len(self.traj_pairs):
                raise Exception(f"Not feasible trajectories in dataset '{cfg.trajectory_data_path}'!")
            
            if self.verbose:
                print(f"Starting enviroment with guiding on {len(self.traj_pairs)} trajectories.")
            
        self.sim = MjSim(self.mujoco_xml, self.tau_sim,
                         interpolate=self.interpolate_actions, joints_are_same_as_ctrl=self.joints_are_same_as_ctrl)
        self.sim.setupRenderer(camera=cfg.sim.camera)
        
        state = self.sim.getState()
        state_n = state[1].shape[0]  # qpos
        self.state_dim = state_n
        if self.use_vel:
            state_n += state[2].shape[0]  # qvel
        if self.goal_conditioning:
            state_n += state[1].shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_n,), dtype=np.float32)
        
        if self.stepsize != -1:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0]
            max_ctrl = ctrl_ranges[:, 1]
        
        ctrl_dim = self.sim.data.ctrl.shape[0]
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(ctrl_dim,), dtype=np.float32)
        
        if self.use_vision:
            self.camera_filter = cfg.camera_filter

    def getState(self) -> np.ndarray:
        time_, qpos, qvel, ctrl = self.sim.getState()
        
        if self.use_vision:
            img = self.sim.renderImg()
            if self.camera_filter == "none":
                self.state = img.astype(np.float32).flatten() / 255
                raise Exception("No vision backbone implemented yet! Refusing to feed full image to model.")
            elif self.camera_filter == "blue_ball_mask":
                _, mask = extract_ball_from_img(img, self.verbose-1)
                self.state = mask
                raise Exception("No vision backbone implemented yet! Refusing to feed full image to model.")
            elif self.camera_filter == "blue_ball_params":
                ball_params, _ = extract_ball_from_img(img, self.verbose-1)
                self.state = ball_params
            else:
                raise Exception(f"Camera filter {self.camera_filter} not implemented yet!")
        else:
            self.state = qpos
            if self.use_vel:
                self.state = np.concatenate((self.state, qvel))
            
        if self.goal_conditioning:
            self.state = np.concatenate((self.state, self.target_state[:self.state_dim]))

        self.last_time = time_
        self.last_ctrl = ctrl
        return self.state

    def reset(self, *, seed: int=None, options: dict=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        # TODO: manually set start and end nodes through options. What if no path?

        # Choose start and end configurations
        if self.guiding:
            traj_idx = np.random.randint(0, len(self.traj_pairs))
            s_cfg_idx = self.traj_pairs[traj_idx][0]
            e_cfg_idx = self.traj_pairs[traj_idx][1]
        
        else:
            s_cfg_idx = self.start_config_idx if self.start_config_idx != -1 else np.random.randint(0, self.config_count)
            e_cfg_idx = self.end_config_idx if self.end_config_idx != -1 else randint_excluding(0, self.config_count, s_cfg_idx)
            
        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx}
        
        self.target_state = self.stable_configs["qpos"][e_cfg_idx]
        # if self.use_vel:
        #     self.target_state = np.concatenate((self.target_state, np.zeros_like(self.sim.data.qvel)))
        
        # Reset simulation state
        self.sim.pushConfig(
            self.stable_configs["qpos"][s_cfg_idx],
            self.stable_configs["ctrl"][s_cfg_idx]
        )
        
        self.iter = 0
        
        # Load guiding trajectory
        if self.guiding and (not len(self.guiding_path) or self.end_config_idx == -1):
            tree = self.trees[s_cfg_idx]
            node = tree[self.traj_end_nodes[traj_idx]]

            # Build guiding trajectory from tree
            self.guiding_path = []
            while True:
                state = node["state"][1]
                # if self.use_vel:
                #     state = np.concatenate((state, node["state"][2]))
                self.guiding_path.append(state)
                if node["parent"] == -1: break
                node = tree[node["parent"]]
            
            self.guiding_path.reverse()

            self.max_steps = int(len(self.guiding_path) * 1.5)
            info["guiding_traj_len"] = len(self.guiding_path)
        
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        return self.getState(), info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if self.actions_noise_sigma != -1:
            action += np.random.randn(action.shape[-1]) * self.actions_noise_sigma
        
        if self.stepsize != -1:
            action += self.last_ctrl
        
        self.sim.step(self.tau_action, action)
        self.getState()
        self.iter += 1

        ### Reward Computation ###
        # Distance to target state
        eval_state = self.state[:self.target_state.shape[0]]
        
        if self.iter < len(self.guiding_path):
            self.reward = .0
        else:
            e = (eval_state - self.target_state) * self.q_mask
            r = -0.1 * np.sqrt(e.T @ e)
            
        # Distance to guiding path
        if self.guiding and self.iter < len(self.guiding_path):
            guiding_step = self.guiding_path[self.iter]
            e = (eval_state - guiding_step) * self.q_mask
            r = np.max((np.sqrt(e.T @ e) * -0.1, -2))
            self.reward += r

        truncated = self.iter >= self.max_steps
        terminated = truncated
        info = {}

        return self.state, self.reward, terminated, truncated, info

    def render(self, mode: str="") -> np.ndarray:
        # Separate scene renderer from model vision

        print("Iter: ", self.iter, "Reward: ", self.reward)

        if mode:
            img = self.sim.renderImg(mode)
        else:
            img = self.sim.renderImg()

        return img
