import h5py
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.env.mujoco_warp_sim import MjSim


class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()
        self.verbose = cfg.get("verbose", 0)
        self.max_steps_default = cfg.max_steps
        
        # Sim interface
        self.sim = MjSim(cfg.sim_interface)
        self.sim_count = cfg.sim_interface.parallel_sims
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
        
        self.stable_qpos = self.original_stable_configs["qpos"][:]
        self.stable_ctrl = self.original_stable_configs["ctrl"][:]
        
        self.original_stable_configs_full = []
        for i in tqdm(range(self.config_count), total=self.config_count):
            self.sim.pushConfig(
                self.original_stable_configs["qpos"][i],
                self.original_stable_configs["ctrl"][i]
            )
            state_vec = self.sim.getCustomStateScaled()[0]
            self.original_stable_configs_full.append(state_vec)
        self.original_stable_configs_full = np.array(self.original_stable_configs_full)
        
        self.iter = np.zeros((self.sim_count,))
        self.target_state = np.zeros((self.sim_count, self.original_stable_configs_full.shape[1]))

        if self.use_csrl:
            self.originals_kd_tree = KDTree(self.original_stable_configs_full)
        
        # Define observation space
        state = self.sim.getCustomState()
        state_dim = state.shape[1] * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define action space
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            min_ctrl = -1.0 * self.stepsize
            max_ctrl = self.stepsize
        else:
            ctrl_ranges = self.sim.model.actuator_ctrlrange
            min_ctrl = ctrl_ranges[:, 0].astype(np.float32)
            max_ctrl = ctrl_ranges[:, 1].astype(np.float32)
        
        ctrl_dim = self.sim.data.ctrl.shape[1]
        self.action_space = spaces.Box(low=min_ctrl, high=max_ctrl, shape=(ctrl_dim,), dtype=np.float32)

        self._cost_buf = np.empty(self.sim_count, dtype=np.float32)
        self._e_buf    = np.empty((self.sim_count, self.original_stable_configs_full.shape[1]), dtype=np.float32)
        self.d_t = np.zeros((self.sim_count,), dtype=np.float32)

    def reset(self, done=None, *, seed: int=None, options: dict={}) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        np.random.seed(seed)

        if "alpha" in options:
            self.schedule_alpha = options["alpha"]
            if self.verbose > 1:
                print("Current alpha: ", self.schedule_alpha)

        if done is None:
            done = np.ones(self.sim_count, dtype=bool)
        reset_idx = np.where(done)[0]
        n_reset = len(reset_idx)

        if n_reset == 0:
            eval_state = self.sim.getCustomStateScaled()
        
            state = self.sim.getCustomState()
            np.subtract(eval_state, self.target_state, out=self._e_buf)
            state = np.concatenate((state, self._e_buf), axis=1)
        
            return state, {}

        # Choose start and end configurations
        s_cfg_idx = np.random.randint(0, self.config_count, (n_reset,))
        e_cfg_idx = np.random.randint(0, self.config_count, (n_reset,))

        if self.use_csrl:
            new_s_cfg_idx = []
            for i in range(n_reset):
                t = self.schedule_alpha * (1. - np.random.uniform(0, 1))
                query = (
                    self.original_stable_configs_full[s_cfg_idx[i]] * t +
                    self.original_stable_configs_full[e_cfg_idx[i]] * (1. - t)
                )
                query = query.reshape(1, -1)
                _, ind = self.originals_kd_tree.query(query, k=1)
                new_s_cfg_idx.append(ind[0][0])
            s_cfg_idx = new_s_cfg_idx

        start_qpos = self.stable_qpos[s_cfg_idx]
        start_ctrl = self.stable_ctrl[s_cfg_idx]
        self.sim.pushConfig(start_qpos, start_ctrl, reset_idx)

        self.max_steps = np.clip(self.schedule_alpha, 0.1, 1.0) * self.max_steps_default
        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx, "reset_idx": reset_idx}
        self.target_state[reset_idx] = self.original_stable_configs_full[e_cfg_idx]
        self.iter[reset_idx] = 0

        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        eval_state = self.sim.getCustomStateScaled()
        
        state = self.sim.getCustomState()
        np.subtract(eval_state, self.target_state, out=self._e_buf)
        state = np.concatenate((state, self._e_buf), axis=1)
        
        self.d_t[reset_idx] = 0
        
        return state, info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            ctrl_np = self.sim.data.ctrl.numpy()
            np.add(action, ctrl_np, out=action)
        
        self.sim.step(
            self.tau_action,
            action
        )
        self.iter += 1

        ### Reward Computation ###
        eval_state = self.sim.getCustomStateScaled()
        
        np.subtract(eval_state, self.target_state, out=self._e_buf)
        np.sum(self._e_buf**2, axis=1, out=self._cost_buf)
        np.sqrt(self._cost_buf, out=self._cost_buf)

        goal_reached = self._cost_buf < self.min_cost
        
        d_t1 = np.clip(1.0 - self._cost_buf / (self.min_cost * 10.0), 0.0, 1.0)
        rewards = d_t1 - self.d_t + goal_reached.astype(np.float32)
        self.d_t = d_t1
        
        # Sparse
        # rewards = goal_reached.astype(np.float32)

        truncated = np.full((self.sim_count,), self.iter >= self.max_steps, dtype=bool)
        terminated = goal_reached

        info = {
            "frames": [],
            "states": [],
            "ctrls": [],
            "goal_reached": goal_reached.astype(np.float32),
            "reward": rewards
        }

        ### CSRL step ###
        if self.use_csrl:
            self.schedule_buffer += 1
            if self.schedule_buffer >= self.schedule_alpha_block:
                self.schedule_buffer = 0
                self.schedule_alpha += self.schedule_alpha_step
                if self.schedule_alpha > 1.0:
                    self.schedule_alpha = 1.0
    
        state = self.sim.getCustomState()
        state = np.concatenate((state, self._e_buf), axis=1)
        return state, rewards, terminated, truncated, info

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
