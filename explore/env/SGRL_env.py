import h5py
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.utils.mj import geom_names2ids
from explore.env.mujoco_warp_sim import MjSim


class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()
        self.verbose = cfg.get("verbose", 0)
        self.max_steps_default = cfg.max_steps
        self.sparse_reward = cfg.get("sparse_reward", True)
        
        # Sim interface
        self.sim = MjSim(cfg.sim_interface)
        self.sim_count = cfg.sim_interface.parallel_sims
        self.min_cost = cfg.min_cost
        self.stepsize = cfg.stepsize
        self.tau_action = cfg.tau_action
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize, dtype=np.float32)
        
        # State info
        self.q = cfg.q
        self.q_dot = cfg.q_dot
        self.q_obj_dot = cfg.q_obj_dot
        self.P = geom_names2ids(cfg.P, self.sim.mj_model)
        self.G = geom_names2ids(cfg.G, self.sim.mj_model)

        self.obs_pos_scale = cfg.get("obs_pos_scale", 1.0)
        self.obs_vel_scale = cfg.get("obs_vel_scale", 0.1)
        self.obs_ref_err_scale = cfg.get("obs_ref_err_scale", 10.0)

        self.q_weight = cfg.q_weight

        # SGRL
        self.use_csrl = cfg.use_csrl
        self.schedule_alpha_step = 1. / (cfg.schedule_alpha_end_step / cfg.schedule_alpha_block)
        self.schedule_alpha_block = cfg.schedule_alpha_block
        self.schedule_alpha = self.schedule_alpha_step
        self.schedule_buffer = 0
        
        self.stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.stable_configs["qpos"].shape[0]
        if self.verbose:
            print("Total configs in h5: ", self.config_count)
        
        self.stable_qpos = self.stable_configs["qpos"][:]
        self.stable_ctrl = self.stable_configs["ctrl"][:]
        
        self.all_G_star = []
        self.phi_stable_configs = []
        for i in tqdm(range(self.config_count), total=self.config_count):
            self.sim.pushConfig(
                self.stable_configs["qpos"][i],
                self.stable_configs["ctrl"][i]
            )
            
            self.sim.gen_numpy_dict()
            self.sim.numpy_dict

            q = self.sim.numpy_dict["qpos"][0, self.q[0]:self.q[1]]
            G = self.sim.numpy_dict["geom_xpos"][0, self.G, :].reshape(-1)
            phi = np.concatenate([q * self.q_weight, G])
            
            self.all_G_star.append(G)
            self.phi_stable_configs.append(phi)

        self.all_G_star = np.array(self.all_G_star)
        self.phi_stable_configs = np.array(self.phi_stable_configs)
        
        self.iter = np.zeros((self.sim_count,))
        self.G_star = np.zeros((self.sim_count, self.all_G_star.shape[1]))

        if self.use_csrl:
            self.originals_kd_tree = KDTree(self.phi_stable_configs)
        
        # Define observation space
        state = self.get_state()
        state_dim = state.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define action space
        ctrl_dim = self.sim.data.ctrl.shape[1]
        self.action_space = spaces.Box(low=-1, high=-1, shape=(ctrl_dim,), dtype=np.float32)

        self._cost_buf = np.empty(self.sim_count, dtype=np.float32)
        self.d_t = np.zeros((self.sim_count,), dtype=np.float32)
        
        self.render = False
    
    def get_state(self) -> np.ndarray:
        self.sim.gen_numpy_dict()
        self.sim.numpy_dict

        q = self.sim.numpy_dict["qpos"][:, self.q[0]:self.q[1]]
        q_dot = self.sim.numpy_dict["qvel"][:, self.q_dot[0]:self.q_dot[1]]
        q_obj_dot = self.sim.numpy_dict["qvel"][:, self.q_obj_dot:self.q_obj_dot+6]
        r = self.sim.numpy_dict["ctrl"]
        P = self.sim.numpy_dict["geom_xpos"][:, self.P, :].reshape(self.sim.nworld, -1)
        G = self.sim.numpy_dict["geom_xpos"][:, self.G, :].reshape(self.sim.nworld, -1)
        
        state = np.concatenate([
            q * self.obs_pos_scale,
            q_dot * self.obs_vel_scale,
            q_obj_dot * self.obs_vel_scale,
            (r - q) * self.obs_ref_err_scale,
            P * self.obs_pos_scale,
            (self.G_star - G) * self.obs_pos_scale
        ], axis=1)
        
        return state

    def reset(self, done=None, *, seed: int=None, options: dict={}) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if "alpha" in options:
            self.schedule_alpha = options["alpha"]
            if self.verbose > 0:
                print("Current alpha: ", self.schedule_alpha)
        
        self.render = "render" in options and options["render"]

        if done is None:
            done = np.ones(self.sim_count, dtype=bool)
        reset_idx = np.where(done)[0]
        n_reset = len(reset_idx)

        if n_reset == 0:
            # TODO: Maybe avoid re-computing the state for sims that have not been reset.
            state = self.get_state()
            return state, {}

        # Choose start and end configurations
        s_cfg_idx = np.random.randint(0, self.config_count, (n_reset,))
        e_cfg_idx = np.random.randint(0, self.config_count, (n_reset,))

        if self.use_csrl:
            new_s_cfg_idx = []
            for i in range(n_reset):
                if "sample_uniform" in options and not options["sample_uniform"]:
                    t = self.schedule_alpha
                else:
                    t = self.schedule_alpha * (1. - np.random.uniform(0, 1))
                query = (
                    self.phi_stable_configs[s_cfg_idx[i]] * t +
                    self.phi_stable_configs[e_cfg_idx[i]] * (1. - t)
                )
                query = query.reshape(1, -1)
                _, ind = self.originals_kd_tree.query(query, k=1)
                new_s_cfg_idx.append(ind[0][0])
            s_cfg_idx = new_s_cfg_idx

        start_qpos = self.stable_qpos[s_cfg_idx]
        start_ctrl = self.stable_ctrl[s_cfg_idx]

        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx, "reset_idx": reset_idx}
        if self.render:
            info["goal_frame"] = self.sim.render_state(self.stable_qpos[e_cfg_idx[0]])
        self.sim.pushConfig(start_qpos, start_ctrl, reset_idx)

        self.max_steps = np.clip(self.schedule_alpha, 0.1, 1.0) * self.max_steps_default
        self.G_star[reset_idx] = self.all_G_star[e_cfg_idx]
        self.iter[reset_idx] = 0

        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        state = self.get_state()
        
        self.d_t[reset_idx] = 0
        
        return state, info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            ctrl_np = self.sim.data.ctrl.numpy()
            ctrl_target = action * self.stepsize + ctrl_np
        else:
            ctrl_target = action.copy()
        
        frames = self.sim.step(
            self.tau_action,
            ctrl_target,
            render=self.render
        )
        state = self.get_state()
        self.iter += 1

        ### Reward Computation ###
        np.sum(state[:, -self.all_G_star.shape[1]:]**2, axis=1, out=self._cost_buf)
        np.sqrt(self._cost_buf, out=self._cost_buf)

        goal_reached = self._cost_buf < self.min_cost
        
        if self.sparse_reward:
            rewards = goal_reached.astype(np.float32)
        
        else:
            d_t1 = np.clip(1.0 - self._cost_buf / (self.min_cost * 10.0), 0.0, 1.0)
            rewards = d_t1 - self.d_t + goal_reached.astype(np.float32)
            self.d_t = d_t1

        terminated = goal_reached
        truncated = np.full((self.sim_count,), self.iter >= self.max_steps, dtype=bool)

        info = {
            "frames": frames,
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
    
        return state, rewards, terminated, truncated, info
