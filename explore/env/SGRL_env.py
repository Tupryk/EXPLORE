import os
import h5py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, OmegaConf, ListConfig

from explore.env.mujoco_sim import MjSim
from explore.utils.mj import get_model_quaternions
from explore.utils.utils import randint_excluding, extract_balls_mask
from explore.datasets.utils import load_trees, cost_computation_on_states, get_diverse_paths


class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()

        self.use_schedule = cfg.use_schedule
        self.schedule_alpha_step = 1. / cfg.schedule_alpha_end_step
        self.schedule_alpha = 0.
        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        self.mujoco_xml = cfg.sim.mujoco_xml
        
        self.stepsize = cfg.stepsize
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize, dtype=np.float32)
        self.max_steps_default = cfg.max_steps  # Gets overwriten if use_guiding = True
        
        self.actions_noise_sigma = cfg.actions_noise_sigma
        self.use_vel = cfg.use_vel
        self.guiding = cfg.guiding
        self.reward = None
        
        self.use_vision = cfg.use_vision
        assert (self.use_vision and not self.use_vel) or (not self.use_vision)

        # Setup sim
        self.start_config_idx = cfg.start_config_idx
        self.end_config_idx = cfg.target_config_idx
        self.goal_conditioning = cfg.goal_conditioning
        if not self.goal_conditioning and cfg.target_config_idx == -1:
            raise Exception("Setting many goals but no goal conditioning!")
        # if self.goal_conditioning and cfg.target_config_idx != -1:
        #     raise Exception("Using goal conditioning but only a single goal!")
        
        self.original_stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.original_stable_configs["q"].shape[0]
        
        self.verbose = cfg.verbose

        config_path = os.path.join(cfg.trajectory_data_path, ".hydra/config.yaml")
        trees_cfg = OmegaConf.load(config_path)
        self.q_mask = np.array(trees_cfg.RRT.q_mask)
        self.min_cost = trees_cfg.RRT.min_cost
        self.dataset_tau_action = trees_cfg.RRT.sim.tau_action
        self.time_scaling = np.ceil(self.dataset_tau_action / self.tau_action) if cfg.time_scaling == -1 else cfg.time_scaling

        tree_dataset = os.path.join(cfg.trajectory_data_path, "trees")
        self.trees, _, _ = load_trees(tree_dataset)

        self.sim = MjSim(self.mujoco_xml, self.tau_sim,
            interpolate=self.interpolate_actions, joints_are_same_as_ctrl=self.joints_are_same_as_ctrl)
        self.sim.setupRenderer(cfg.render_w, cfg.render_h, camera=cfg.sim.camera)
        self.model_quats = get_model_quaternions(self.sim.model)
        
        if self.guiding == "StaGE":

            self.paths, self.traj_pairs = get_diverse_paths(
                self.trees,
                self.min_cost,
                self.q_mask,
                trees_cfg.RRT.path_diff_thresh,
                cached_folder=cfg.trajectory_data_path,
                scene_quats=self.model_quats
            )
            if self.verbose > 2:
                input("Sample trajectories loaded. Press enter to continue.")
            
            if not len(self.traj_pairs):
                raise Exception(f"Not feasible trajectories in dataset '{cfg.trajectory_data_path}'!")
            
            if self.verbose > 0:
                print(f"Starting enviroment with guiding on {len(self.traj_pairs)} trajectories with average length {sum(len(p) for p in self.paths)/len(self.traj_pairs)}.")

        elif self.guiding == "SGRL":
            self.originals_kd_tree = KDTree(self.original_stable_configs["q"])
        
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
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0].astype(np.float32)
            max_ctrl = ctrl_ranges[:, 1].astype(np.float32)
        
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(self.ctrl_dim,), dtype=np.float32)
        
        if self.use_vision:
            self.camera_filter = cfg.camera_filter
        
    def getState(self) -> np.ndarray:
        
        self.sim_state = self.sim.getState()
        time_, qpos, qvel, ctrl, _, _ = self.sim_state
        
        if self.use_vision:
            
            img = self.sim.renderImg()
            prio = qpos[:self.ctrl_dim]

            if self.goal_conditioning:
                prio = np.concatenate((prio, self.target_state))
            
            if not self.camera_filter or self.camera_filter == "none":
                pass
            
            elif self.camera_filter == "balls_mask":
                img = extract_balls_mask(img, self.verbose-1)
            
            else:
                raise Exception(f"Camera filter {self.camera_filter} not implemented yet!")
            
            self.state = {
                "image": img.astype(np.uint8),
                "proprio": prio,
            }
            
        else:
            self.state = qpos
            if self.use_vel:
                self.state = np.concatenate((self.state, qvel))
            
            if self.goal_conditioning:
                self.state = np.concatenate((self.state, self.target_state))

        self.last_time = time_
        self.last_ctrl = ctrl
        return self.state

    def reset(self, *, seed: int=None, options: dict={}):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.eval_view = "" if not "eval_view" in options else options["eval_view"]

        if "alpha" in options:
            self.schedule_alpha = options["alpha"]

        if self.verbose > 1:
            print("Current alpha: ", self.schedule_alpha)

        # Choose start and end configurations
        if self.guiding == "StaGE":
            
            if "traj_pair" in options and options["traj_pair"] != (-1, -1):  # Only for eval!!!

                tp = options["traj_pair"]
                if tp in self.traj_pairs:
                    traj_idx = self.traj_pairs.index(tp)
                else:
                    print("Running unknown trajectory. Very scary!!")
            
            else:
                traj_idx = np.random.randint(0, len(self.traj_pairs))
            
            s_cfg_idx = self.traj_pairs[traj_idx][0]
            e_cfg_idx = self.traj_pairs[traj_idx][1]

            guiding_path = self.paths[traj_idx]
            guiding_path_len = len(guiding_path)
            self.max_steps = int(len(guiding_path) * 1.5) * self.time_scaling

            node_idx = int((guiding_path_len-1) * (1.0 - self.schedule_alpha))
            if node_idx >= guiding_path_len-1: node_idx = guiding_path_len-2
            self.max_steps = max(int(len(guiding_path[node_idx:]) * 1.5) * self.time_scaling, 20)
            self.sim.setState(*guiding_path[node_idx])

        else:
            s_cfg_idx = self.start_config_idx if self.start_config_idx != -1 else np.random.randint(0, self.config_count)
            e_cfg_idx = self.end_config_idx if self.end_config_idx != -1 else randint_excluding(0, self.config_count, s_cfg_idx)
            
            if self.guiding == "SGRL":
                query = (
                    self.original_stable_configs["q"][s_cfg_idx] * self.schedule_alpha +
                    self.original_stable_configs["q"][e_cfg_idx] * (1. - self.schedule_alpha)
                )
                query = query.reshape(1, -1)
                _, ind = self.originals_kd_tree.query(query, k=1)
                s_cfg_idx = ind[0][0]
        
            start_qpos = self.original_stable_configs["q"][s_cfg_idx]
            start_ctrl = self.original_stable_configs["ctrl"][s_cfg_idx]

            self.sim.pushConfig(start_qpos, start_ctrl)
            self.max_steps = self.max_steps_default
            
        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx}
        self.target_state = self.original_stable_configs["q"][e_cfg_idx]
        
        self.iter = 0
        
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        return self.getState(), info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if self.actions_noise_sigma != -1:
            action += np.random.randn(action.shape[-1]) * self.actions_noise_sigma
        
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            action += self.last_ctrl
        self.last_ctrl = action
        
        frames, ss, cs = self.sim.step(
            self.tau_action,
            action,
            view=self.eval_view,
            log_all=bool(self.eval_view)
        )
        self.getState()
        self.iter += 1

        ### Reward Computation ###
        eval_state = self.sim_state[1]
        
        cost = cost_computation_on_states(eval_state, self.target_state, self.q_mask, scene_quat_indices=self.model_quats)
        
        if cost < self.min_cost:
            goal_reached_reward = 1.0 
            truncated = True
        else:
            goal_reached_reward = 0.0 
            truncated = False

        self.reward = goal_reached_reward

        if self.iter > self.max_steps:
            truncated = True
        
        terminated = truncated
        info = {
            "frames": frames,
            "states": ss,
            "ctrls": cs,
            "reward": self.reward
        }

        self.schedule_alpha += self.schedule_alpha_step
        if self.schedule_alpha > 1.0:
            self.schedule_alpha = 1.0

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
