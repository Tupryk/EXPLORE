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
        self.use_csrl = cfg.get("use_csrl", True)
        self.schedule_alpha_step = 1. / (cfg.schedule_alpha_end_step / cfg.schedule_alpha_block)
        self.schedule_alpha_block = cfg.schedule_alpha_block
        self.schedule_alpha = self.schedule_alpha_step
        self.schedule_buffer = 0
        
        self.expand_manifold = cfg.get("expand_manifold", False)
        self.max_manifold = int(cfg.get("max_manifold", 1e6))
        self.manifold_idx = 0
        
        self.object_diffusion = cfg.get("object_diffusion", False)
        
        assert not (self.expand_manifold and self.object_diffusion)
        assert not (self.use_csrl and self.object_diffusion)
        
        self.stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.stable_configs["qpos"].shape[0]
        if self.verbose:
            print("Total configs in h5: ", self.config_count)
        
        if self.expand_manifold:
            self.manifold_qpos = np.zeros((self.max_manifold, self.sim.data.qpos.shape[1]))
            self.manifold_qvel = np.zeros((self.max_manifold, self.sim.data.qvel.shape[1]))
            self.manifold_ctrl = np.zeros((self.max_manifold, self.sim.data.ctrl.shape[1]))
            
            self.manifold_qpos[:self.config_count] = self.stable_configs["qpos"][:]
            self.manifold_ctrl[:self.config_count] = self.stable_configs["ctrl"][:]
            
        else:
            self.manifold_qpos = self.stable_configs["qpos"][:]
            self.manifold_qvel = np.zeros((self.config_count, self.sim.data.qvel.shape[1]))
            self.manifold_ctrl = self.stable_configs["ctrl"][:]
        
        self.all_G_star = []
        self.phi_stable_configs = []
        for i in tqdm(range(self.config_count), total=self.config_count):
            self.sim.mj_data.qpos[:] = self.stable_configs["qpos"][i]
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

        if self.use_csrl and not self.object_diffusion:
            if self.expand_manifold:
                self.sds = hnswlib.Index(space="l2", dim=self.phi_stable_configs.shape[1])
                self.sds.init_index(max_elements=self.max_manifold, ef_construction=200, M=16)
                
                self.manifold_idx = self.phi_stable_configs.shape[0]
                self.sds.add_items(self.phi_stable_configs, ids=list(range(0, self.manifold_idx)))
                
            else:
                self.sds = KDTree(self.phi_stable_configs)
        
        # Define observation space
        state, _ = self.get_state()
        state_dim = state.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define action space
        ctrl_dim = self.sim.data.ctrl.shape[1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(ctrl_dim,), dtype=np.float32)

        self._cost_buf = np.empty(self.sim_count, dtype=np.float32)
        self.d_t = np.zeros((self.sim_count,), dtype=np.float32)
        
        self.render = False
    
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
                
                if self.expand_manifold:
                    ind = self.sds.knn_query(query, k=1)
                else:
                    _, ind = self.sds.query(query, k=1)
                new_s_cfg_idx.append(int(ind[0][0]))
            s_cfg_idx = new_s_cfg_idx

        info = {"start_config_idx": s_cfg_idx, "end_config_idx": e_cfg_idx, "reset_idx": reset_idx}
        if self.render and not self.object_diffusion:
            info["goal_frame"] = self.sim.render_state(self.manifold_qpos[e_cfg_idx[0]])
        
        self.sim.setState(
            np.zeros((reset_idx.shape[0],)),
            self.manifold_qpos[s_cfg_idx],
            self.manifold_qvel[s_cfg_idx],
            self.manifold_ctrl[s_cfg_idx],
            reset_idx
        )

        self.max_steps = np.clip(self.schedule_alpha, 0.1, 1.0) * self.max_steps_default
        if self.object_diffusion:
            stds = np.array([2., 2., .3, 1., 1., 1., 1.])
            offsets = np.array([0., 0., 1., 0., 0., 0., 0.])
            
            for i, ri in enumerate(reset_idx):
                noised_cube = np.random.randn(7) * stds + offsets

                t = self.schedule_alpha * (1. - np.random.uniform(0, 1))
                cube_pos = self.manifold_qpos[s_cfg_idx[i]][-7:] * (1. - t) + noised_cube * t
                
                noised_qpos = self.manifold_qpos[s_cfg_idx[i]].copy()
                noised_qpos[-7:] = cube_pos
                self.sim.mj_data.qpos[:] = noised_qpos
                mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

                G = self.sim.mj_data.geom_xpos[self.G, :].reshape(-1)
                self.G_star[ri] = G
            
                if i == 0:
                    info["goal_frame"] = self.sim.render_state(noised_qpos)
        else:
            self.G_star[reset_idx] = self.all_G_star[e_cfg_idx]
        self.iter[reset_idx] = 0

        if self.verbose > 1:
            print(f"Reseting enviroment with start config {s_cfg_idx} and end config {e_cfg_idx}.")

        state, _ = self.get_state()
        
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
        state, state_dict = self.get_state()
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
        
        if self.expand_manifold:
            # TODO: Maybe this should be delayed to when the model has learned something?
            q = self.sim.numpy_dict["qpos"][:, self.q[0]:self.q[1]]
            G = self.sim.numpy_dict["geom_xpos"][:, self.G, :].reshape(self.sim_count, -1)
            phi = np.concatenate([q * self.q_weight, G], axis=1)
            
            indices = (self.manifold_idx + np.arange(self.sim_count)) % (self.max_manifold - self.config_count)
            indices += self.config_count
            self.sds.add_items(phi, ids=indices)
            self.manifold_idx = (self.manifold_idx + self.sim_count) % (self.max_manifold - self.config_count)
            
            self.manifold_qpos[indices] = self.sim.numpy_dict["qpos"][:]
            self.manifold_qvel[indices] = self.sim.numpy_dict["qvel"][:]
            self.manifold_ctrl[indices] = self.sim.numpy_dict["ctrl"][:]

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
