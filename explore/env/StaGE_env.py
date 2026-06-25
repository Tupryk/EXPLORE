import h5py
import mujoco
import numpy as np
from tqdm import tqdm
from gymnasium import spaces
from sklearn.neighbors import KDTree
from omegaconf import DictConfig, ListConfig

from explore.utils.mj import geom_names2ids
from explore.env.mujoco_warp_sim import MjSim


class StaGEEnv():

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

        # State Info
        self.q = cfg.q
        self.q_dot = cfg.q_dot
        self.q_obj_dot = cfg.q_obj_dot
        self.P = geom_names2ids(cfg.P, self.sim.mj_model)
        self.G = geom_names2ids(cfg.G, self.sim.mj_model)

        # Observation Scaling
        self.obs_pos_scale = cfg.get("obs_pos_scale", 1.0)
        self.obs_vel_scale = cfg.get("obs_vel_scale", 0.1)
        self.obs_ref_err_scale = cfg.get("obs_ref_err_scale", 10.0)

        self.q_weight = cfg.q_weight

        # Stable Manifold
        max_manifold_size = cfg.get("max_manifold_size", -1)
        h5py_data = h5py.File(cfg.stable_configs_path, 'r')
        self.manifold_qpos = h5py_data["qpos"]
        self.manifold_ctrl = h5py_data["ctrl"]

        if max_manifold_size != -1:
            self.manifold_qpos = self.manifold_qpos[:max_manifold_size]
        
        self.manifold_size = self.manifold_qpos.shape[0]

        if self.verbose:
            print("Total configs in h5: ", self.manifold_size)
        
        self.all_G_star = []
        self.phi_stable_configs = []
        for i in tqdm(range(self.manifold_size), total=self.manifold_size):
            self.sim.mj_data.qpos[:] = self.manifold_qpos[i]
            mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

            q = self.sim.mj_data.qpos[self.q[0]:self.q[1]]
            G = self.sim.mj_data.geom_xpos[self.G, :].reshape(-1)
            phi = np.concatenate([q * self.q_weight, G])
            
            self.all_G_star.append(G)
            self.phi_stable_configs.append(phi)

        self.all_G_star = np.array(self.all_G_star)
        self.phi_stable_configs = np.array(self.phi_stable_configs)
        
        self.iter = np.zeros((self.sim_count,))
        self.G_star = np.zeros((self.sim_count, self.all_G_star.shape[1]))

        self.sds = KDTree(self.phi_stable_configs)
        
        # StaGE

        # Define observation space
        state = self.get_state()
        state_dim = state.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define action space
        ctrl_dim = self.sim.data.ctrl.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(ctrl_dim,), dtype=np.float32)

        self._cost_buf = np.empty(self.sim_count, dtype=np.float32)
        self.d_t = np.zeros((self.sim_count,), dtype=np.float32)
    
    def get_state(self) -> tuple[np.ndarray, dict]:
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

    def reset(self, options: dict={}) -> tuple[np.ndarray, dict]:
        
        self.buffer = []
        self.nodes = []
        self.closest_nodes = [0 for _ in range(self.manifold_size)]
        
        # Choose start and end configurations
        if "start_idx" in options and options["start_idx"] != -1:
            start_idx = options["start_idx"]
        else:
            start_idx = np.random.randint(0, self.manifold_size)
        target_idx = np.random.randint(0, self.manifold_size)

        info = {"target_config_idx": target_idx}
        
        self.sim.setState(
            np.zeros((1,)),
            self.manifold_qpos[start_idx],
            np.zeros_like(self.sim.mj_data.qvel),
            self.manifold_ctrl[start_idx]
        )

        state = self.get_state()
        
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

        # Add actions to tree
        # For each action do k=32 nearest neighbour search and check for each if
        # the distance was reduced and update the lookup tables for each target
        # on the manifold.
        
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
            "goal_reached": terminated.astype(np.float32), # If true, dataset has to be relabeled
            "reward": rewards
        }

        return state, rewards, terminated, truncated, info
    
    def sample(self) -> tuple:
        return
