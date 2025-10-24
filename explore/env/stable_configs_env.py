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
        self.max_steps_default = cfg.max_steps  # Gets overwriten if use_guiding = True
        
        self.actions_noise_sigma = cfg.actions_noise_sigma
        self.use_vel = cfg.use_vel
        self.guiding = cfg.guiding
        self.guiding_prob = cfg.guiding_prob
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
        self.dataset_tau_action = trees_cfg.RRT.sim.tau_action
        self.time_scaling = np.ceil(self.tau_action / self.dataset_tau_action)

        self.guiding_path = []
        if self.guiding:
            
            tree_dataset = os.path.join(cfg.trajectory_data_path, "trees")
            self.trees, _, _ = load_trees(tree_dataset)
            
            min_costs, top_nodes = generate_adj_map(self.trees, self.q_mask, check_cached=cfg.trajectory_data_path)
            self.traj_pairs, self.traj_end_nodes, _ = get_feasible_paths(
                min_costs, top_nodes, self.start_config_idx, self.end_config_idx, trees_cfg.RRT.min_cost)
            
            if not len(self.traj_pairs):
                raise Exception(f"Not feasible trajectories in dataset '{cfg.trajectory_data_path}'!")
            
            if self.verbose > 0:
                print(f"Starting enviroment with guiding on {len(self.traj_pairs)} trajectories.")
            
        self.sim = MjSim(self.mujoco_xml, self.tau_sim,
                         interpolate=self.interpolate_actions, joints_are_same_as_ctrl=self.joints_are_same_as_ctrl)
        self.sim.setupRenderer(84, 84, camera=cfg.sim.camera)
        
        # Defines observation space
        state = self.sim.getState()
        state_n = state[1].shape[0]  # qpos
        self.state_dim = state_n
        if self.use_vel:
            state_n += state[2].shape[0]  # qvel
        if self.goal_conditioning:
            state_n += state[1].shape[0]
        
        self.ctrl_dim = self.sim.data.ctrl.shape[0]

        if self.use_vision:
            obs_n = self.ctrl_dim + self.state_dim if self.goal_conditioning else self.ctrl_dim
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_n,), dtype=np.float32),
            })

        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_n,), dtype=np.float32)
        
        # Defines action space
        if self.stepsize != -1:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0]
            max_ctrl = ctrl_ranges[:, 1]
        
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(self.ctrl_dim,), dtype=np.float32)
        
        if self.use_vision:
            self.camera_filter = cfg.camera_filter

    def getState(self) -> np.ndarray:
        
        self.sim_state = self.sim.getState()
        time_, qpos, qvel, ctrl = self.sim_state
        
        if self.use_vision:
            
            img = self.sim.renderImg()
            
            if not self.camera_filter or self.camera_filter == "none":
                
                prio = qpos[:self.ctrl_dim]

                if self.goal_conditioning:
                    prio = np.concatenate((prio, self.target_state[:self.state_dim]))

                self.state = {
                    "image": img.astype(np.uint8),
                    "proprio": prio,
                }
            
            elif self.camera_filter == "blue_ball_mask":
                _, mask = extract_ball_from_img(img, self.verbose-1)
                self.state = mask
                raise Exception("Not fully implemented yet!")
            
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

    def reset(self, *, seed: int=None, options: dict={}):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.eval_view = "" if not "eval_view" in options else options["eval_view"]

        # Choose start and end configurations
        self.unknown_path = False
        self.currently_guiding = self.guiding and np.random.random() < self.guiding_prob
        if self.currently_guiding:
            
            if "no_exist_fine" in options and options["no_exist_fine"]:
                s_cfg_idx = options["traj_pair"][0]
                e_cfg_idx = options["traj_pair"][1]
                self.unknown_path = True
                if not (options["traj_pair"] in self.traj_pairs):
                    print("Running unknown trajectory. Very scary!!")

            else:
                if "traj_pair" in options and options["traj_pair"] != (-1, -1):

                    tp = options["traj_pair"]
                    if tp in self.traj_pairs:
                        traj_idx = self.traj_pairs.index(tp)
                    else:
                        raise Exception(f"Trajectory pair '{options['traj_pair']}' not in list! Availible pairs: {self.traj_pairs}")
                
                else:
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
        self.guiding_path = []
        if self.currently_guiding and not self.unknown_path and (not len(self.guiding_path) or self.end_config_idx == -1):
            tree = self.trees[s_cfg_idx]
            node = tree[self.traj_end_nodes[traj_idx]]

            # Build guiding trajectory from tree
            while True:
                state = node["state"][1]
                # if self.use_vel:
                #     state = np.concatenate((state, node["state"][2]))
                self.guiding_path.append(state)
                if node["parent"] == -1: break
                node = tree[node["parent"]]
            
            self.guiding_path.reverse()

            self.max_steps = int(len(self.guiding_path) * 1.5) * self.time_scaling
            info["guiding_traj_len"] = len(self.guiding_path)
        
        else:
            self.max_steps = self.max_steps_default * self.time_scaling
        
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        return self.getState(), info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if self.actions_noise_sigma != -1:
            action += np.random.randn(action.shape[-1]) * self.actions_noise_sigma
        
        if self.stepsize != -1:
            action += self.last_ctrl
        
        frames = self.sim.step(self.tau_action, action, view=self.eval_view)
        self.getState()
        self.iter += 1

        ### Reward Computation ###
        
        eval_state = self.sim_state[1]
        
        goal_reward = 0.0
        if self.iter >= (len(self.guiding_path) * self.time_scaling):
            
            e = (eval_state - self.target_state) * self.q_mask
            
            goal_reward = -1.0 * (e.T @ e)
            goal_reward = np.max((goal_reward, -5.0))
        
        guiding_reward = 0.0
        if self.currently_guiding and not self.unknown_path and self.iter < (len(self.guiding_path) * self.time_scaling):
            
            path_idx = int(np.ceil(self.iter/self.time_scaling))
            guiding_step = self.guiding_path[path_idx]

            e = (eval_state - guiding_step) * self.q_mask
            guiding_reward = -1.0 * (e.T @ e)
            guiding_reward = np.max((guiding_reward, -5.0))

        self.reward = goal_reward * 1.0 + guiding_reward * 0.1

        truncated = self.iter >= self.max_steps
        terminated = truncated
        info = {
            "frames": frames,
            "goal_reward": goal_reward,
            "guiding_reward": guiding_reward
        }

        return self.state, self.reward, terminated, truncated, info

    def render(self, mode: str="", config_idx: int=-1) -> np.ndarray:
        # Separate scene renderer from model vision

        if config_idx != -1:
            current_state = self.sim.getState()
            self.sim.setState(*self.trees[config_idx][0]["state"])

        if mode:
            img = self.sim.renderImg(mode)
        else:
            img = self.sim.renderImg()

        if config_idx != -1:
            self.sim.setState(*current_state)

        return img
