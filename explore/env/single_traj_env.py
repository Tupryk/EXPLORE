import h5py
import mujoco
import hnswlib
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.utils.mj import geom_names2ids
# from explore.env.mujoco_warp_sim import MjSim
from explore.env.mujoco_threaded_sim import MjSim


class SingleTrajEnv(gym.Env):

    def __init__(self, goal_state: np.ndarray, traj: list, cfg: DictConfig, interpolate: bool=True):
        
        super().__init__()
        self.cfg = cfg
        self.traj = traj
        self.goal_state = goal_state
        self.verbose = cfg.get("verbose", 0)
        self.max_steps_default = cfg.max_steps
        self.sparse_reward = cfg.get("sparse_reward", True)
        
        # Sim interface
        self.sim = MjSim(cfg.sim_interface)
        self.sim_count = cfg.sim_interface.parallel_sims
        self.min_cost = cfg.min_cost
        self.stepsize = cfg.stepsize
        self.tau_action = cfg.tau_action
        self.interpolate = interpolate
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

        # Curriculum
        if cfg.use_curriculum:
            self.schedule_alpha_step = 1. / (cfg.schedule_alpha_end_step / cfg.schedule_alpha_block)
            self.schedule_alpha_block = cfg.schedule_alpha_block
            self.schedule_alpha = self.schedule_alpha_step
            self.schedule_buffer = 0
        
        else:
            self.schedule_alpha = 1.
        
        self.iter = np.zeros((self.sim_count,))

        # Define observation space
        state, _ = self.get_state()
        state_dim = state.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define action space
        ctrl_dim = self.sim.mj_data.ctrl.shape[0]
        self.action_space = spaces.Box(low=-1, high=1, shape=(ctrl_dim,), dtype=np.float32)

        self._cost_buf = np.empty(self.sim_count, dtype=np.float32)
        self.d_t = np.zeros((self.sim_count,), dtype=np.float32)

        self.render = False

    def state2G(self, state_dict: dict) -> np.ndarray:
        G = state_dict["geom_xpos"][:, self.G, :].reshape(self.sim.nworld, -1)
        return G
    
    def get_state(self) -> tuple[np.ndarray, dict]:
        self.sim.gen_numpy_dict()
        self.sim.numpy_dict

        q = self.sim.numpy_dict["qpos"][:, self.q[0]:self.q[1]]
        q_dot = self.sim.numpy_dict["qvel"][:, self.q_dot[0]:self.q_dot[1]]
        q_obj_dot = np.concatenate([
            self.sim.numpy_dict["qvel"][:, i:i+6] for i in self.q_obj_dot
        ], axis=1)
        r = self.sim.numpy_dict["ctrl"]
        P = self.sim.numpy_dict["geom_xpos"][:, self.P, :].reshape(self.sim.nworld, -1)
        
        state = np.concatenate([
            q * self.obs_pos_scale,
            q_dot * self.obs_vel_scale,
            q_obj_dot * self.obs_vel_scale,
            (r - q) * self.obs_ref_err_scale,
            P * self.obs_pos_scale
        ], axis=1)
        
        return state, self.sim.numpy_dict

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
            state, _ = self.get_state()
            return state, {}

        # Choose start and end configurations
        uni = np.random.random((reset_idx.shape[0],)) if self.interpolate else np.ones((reset_idx.shape[0],))
        ids = (1. - (uni * self.schedule_alpha)) * len(self.traj)
        ids = [int(id) for id in ids]

        self.sim.setState(
            np.zeros((reset_idx.shape[0],)),
            [self.traj[id]["qpos"] for id in ids],
            [self.traj[id]["qvel"] for id in ids],
            [self.traj[id]["ctrl"] for id in ids],
            reset_idx
        )

        self.max_steps = np.clip(self.schedule_alpha, 0.1, 1.0) if self.cfg.use_curriculum else 1.0
        self.max_steps *= self.max_steps_default

        self.iter[reset_idx] = 0

        state, _ = self.get_state()
        
        self.d_t[reset_idx] = 0

        info = {}
        return state, info

    def step(self, action: np.ndarray):
        
        ### Simulation Step ###
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            ctrl_np = self.sim.data_ctrl.copy()
            ctrl_target = action * self.stepsize + ctrl_np
        else:
            ctrl_target = action.copy()
        
        frames = self.sim.step(
            self.tau_action,
            ctrl_target,
            render=self.render
        )
        state, state_dict = self.get_state()
        self.iter += 1

        ### Reward Computation ###
        np.sum((self.state2G(state_dict) - self.goal_state)**2, axis=1, out=self._cost_buf)
        np.sqrt(self._cost_buf, out=self._cost_buf)

        goal_reached = self._cost_buf < self.min_cost
        
        if self.sparse_reward:
            # rewards = goal_reached.astype(np.float32) - 1.
            rewards = goal_reached.astype(np.float32)
        
        else:
            d_t1 = np.clip(1.0 - self._cost_buf / (self.min_cost * 10.0), 0.0, 1.0)
            rewards = d_t1 - self.d_t + goal_reached.astype(np.float32)
            self.d_t = d_t1

            rewards -= 4e-4 * np.array([qvel @ qvel for qvel in state_dict["qvel"]]).flatten()
        
        terminated = goal_reached
        truncated = np.full((self.sim_count,), self.iter >= self.max_steps)

        info = {
            "frames": frames,
            "states": [],
            "ctrls": [],
            "goal_reached": goal_reached.astype(np.float32),
            "reward": rewards
        }

        if self.cfg.use_curriculum:
            self.schedule_buffer += 1
            if self.schedule_buffer >= self.schedule_alpha_block:
                self.schedule_buffer = 0
                self.schedule_alpha += self.schedule_alpha_step
                if self.schedule_alpha > 1.0:
                    self.schedule_alpha = 1.0
    
        return state, rewards, terminated, truncated, info
