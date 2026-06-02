import h5py
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.env.mujoco_sim import MjSim


class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()
        self.verbose = cfg.get("verbose", 0)
        self.max_steps_default = cfg.max_steps
        
        # Sim interface
        self.sim = MjSim(cfg.sim_interface)
        self.min_cost = cfg.min_cost
        self.stepsize = cfg.stepsize
        self.tau_action = cfg.tau_action
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize, dtype=np.float32)
        
        # SGRL
        self.use_csrl = cfg.use_csrl
        self.schedule_alpha_step = 1. / (cfg.schedule_alpha_end_step / cfg.schedule_alpha_block)
        self.schedule_alpha_block = cfg.schedule_alpha_block
        self.schedule_alpha = self.schedule_alpha_step
        self.schedule_buffer = 0
        
        self.original_stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.original_stable_configs["qpos"].shape[0]
        if self.verbose:
            print("Total configs in h5: ", self.config_count)
            
        self.original_stable_configs_full = []
        for i in tqdm(range(self.config_count), total=self.config_count):
            self.sim.pushConfig(self.original_stable_configs["qpos"][i], self.original_stable_configs["ctrl"][i])
            state_vec = self.sim.getCustomStateScaled()
            self.original_stable_configs_full.append(state_vec)

        if self.use_csrl:
            self.originals_kd_tree = KDTree(self.original_stable_configs_full)
        
        # Define observation space
        state = self.sim.getCustomState()
        state_n = state.shape[0] * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_n,), dtype=np.float32)
        
        # Define action space
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0].astype(np.float32)
            max_ctrl = ctrl_ranges[:, 1].astype(np.float32)
        
        ctrl_dim = self.sim.data.ctrl.shape[0]
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(ctrl_dim,), dtype=np.float32)
        
    def getState(self) -> np.ndarray:
        state = self.sim.getCustomState()
        state = np.concatenate((state, self.goal_condition))
        return state

    def reset(self, *, seed: int=None, options: dict={}) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        np.random.seed(seed)

        self.eval_view = "" if not "eval_view" in options else options["eval_view"]

        if "alpha" in options:
            self.schedule_alpha = options["alpha"]

        if self.verbose > 1:
            print("Current alpha: ", self.schedule_alpha)

        # Choose start and end configurations
        s_cfg_idx = np.random.randint(0, self.config_count)
        e_cfg_idx = np.random.randint(0, self.config_count)
        
        if self.use_csrl:
            query = (
                self.original_stable_configs_full[s_cfg_idx] * self.schedule_alpha +
                self.original_stable_configs_full[e_cfg_idx] * (1. - self.schedule_alpha)
            )
            query = query.reshape(1, -1)
            _, ind = self.originals_kd_tree.query(query, k=1)
            s_cfg_idx = ind[0][0]
    
        start_qpos = self.original_stable_configs["qpos"][s_cfg_idx]
        start_ctrl = self.original_stable_configs["ctrl"][s_cfg_idx]

        self.sim.pushConfig(start_qpos, start_ctrl)
        self.max_steps = np.clip(self.schedule_alpha, 0.1, 1.0) * self.max_steps_default
            
        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx}
        self.target_state = self.original_stable_configs_full[e_cfg_idx]
        self.goal_condition = self.original_stable_configs_full[e_cfg_idx]
        
        self.iter = 0
        
        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        return self.getState(), info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            action += np.copy(self.sim.data.ctrl)
        
        frames, ss, cs = self.sim.step(
            self.tau_action,
            action,
            view=self.eval_view,
            log_all=bool(self.eval_view)
        )
        self.iter += 1

        ### Reward Computation ###
        eval_state = self.sim.getCustomStateScaled()
        
        e = eval_state - self.target_state
        cost = e.T @ e
        if cost < self.min_cost:
            goal_reached_reward = 1.0
            terminated = True
        else:
            goal_reached_reward = 0.0
            terminated = False

        self.reward = goal_reached_reward

        ### Logging ###
        truncated = self.iter >= self.max_steps
        
        terminated = truncated or terminated
        info = {
            "frames": frames,
            "states": ss,
            "ctrls": cs,
            "reward": self.reward
        }

        ### CSRL step ###
        if self.use_csrl:
            self.schedule_buffer += 1
            if self.schedule_buffer >= self.schedule_alpha_block:
                self.schedule_buffer = 0
                self.schedule_alpha += self.schedule_alpha_step
                if self.schedule_alpha > 1.0:
                    self.schedule_alpha = 1.0

        return self.getState(), self.reward, terminated, truncated, info

    def render(self, mode: str="", config_idx: int=-1) -> np.ndarray:
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
