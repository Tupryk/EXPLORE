import os
import h5py
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from omegaconf import DictConfig

from explore.env.mujoco_sim import MjSim
from explore.utils.utils import randint_excluding
from explore.datasets.utils import cost_computation


def getFeasibleTransitionPairs(
    trees: list[list[dict]], feasible_thresh: float=5e-2,
    start_idx: int=-1, end_idx: int=-1) -> tuple[list[tuple[int, int]], list[int]]:
    
    top_nodes = []
    min_costs = []
    
    tree_count = len(trees)
    for i in range(tree_count):

        tree_min_costs = [float("inf") for _ in range(tree_count)]
        tree_top_nodes = [-1 for _ in range(tree_count)]
        
        for n, node in enumerate(trees[i]):
            for j in range(tree_count):
                cost = cost_computation(node, trees[j][0])
                if cost < tree_min_costs[j]:
                    tree_min_costs[j] = cost
                    tree_top_nodes[j] = n
        
        top_nodes.append(tree_top_nodes)
        min_costs.append(tree_min_costs)
    
    feasible_pairs = []
    end_nodes = []
    for s in range(tree_count):
        for e in range(tree_count):
            if (s != e and min_costs[s][e] <= feasible_thresh
                and (start_idx == -1 or start_idx == s) and (end_idx == -1 or end_idx == e)):
                feasible_pairs.append((s, e))
                end_nodes.append(top_nodes[s][e])

    return feasible_pairs, end_nodes

class StableConfigsEnv(gym.Env):

    def __init__(self, cfg: DictConfig):
        
        super().__init__()

        self.tau_sim = cfg.tau_sim
        self.tau_action = cfg.tau_action
        self.interpolate_actions = cfg.interpolate_actions
        self.mujoco_xml = cfg.mujoco_xml
        
        self.stepsize = cfg.stepsize
        self.max_steps = cfg.max_steps  # Gets overwriten if use_guiding = True
        
        self.actions_noise_sigma = cfg.actions_noise_sigma
        self.use_vel = cfg.use_vel
        self.guiding = cfg.guiding
        self.reward = None

        # Setup sim
        self.start_config_idx = cfg.start_config_idx
        self.end_config_idx = cfg.target_config_idx
        self.goal_conditioning = cfg.goal_conditioning
        if not self.goal_conditioning and cfg.target_config_idx == -1:
            raise Exception("Setting many goals but no goal conditioning!")
        
        self.stable_configs = h5py.File(cfg.stable_configs_path, 'r')
        self.config_count = self.stable_configs["qpos"].shape[0]
        
        self.verbose = cfg.verbose

        self.guiding_path = []
        if self.guiding:
            
            tree_dataset = os.path.join(cfg.trajectory_data_path, "trees")
            self.trees: list[list[dict]] = []

            for i in range(self.config_count):
                data_path = os.path.join(tree_dataset, f"tree_{i}.pkl")
                with open(data_path, "rb") as f:
                    tree: list[dict] = pickle.load(f)
                    self.trees.append(tree)

            self.traj_pairs, self.traj_end_nodes = getFeasibleTransitionPairs(
                self.trees, cfg.error_thresh, self.start_config_idx, self.end_config_idx)
            
            if not len(self.traj_pairs):
                raise Exception(f"Not feasible trajectories in dataset '{cfg.trajectory_data_path}'!")
            
            if self.verbose:
                print(f"Starting enviroment with guiding on {len(self.traj_pairs)} trajectories.")
            
        self.sim = MjSim(self.mujoco_xml, self.tau_sim, interpolate=self.interpolate_actions)
        
        state = self.sim.getState()
        state_n = state[1].shape[0]  # qpos
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

    def getState(self) -> np.ndarray:
        time_, qpos, qvel, ctrl = self.sim.getState()
        self.state = qpos
        
        if self.use_vel:
            self.state = np.concatenate((self.state, qvel))
        
        if self.goal_conditioning:
            self.state = np.concatenate((self.state, self.target_state))
        
        self.last_time = time_
        self.last_ctrl = ctrl
        return self.state

    def reset(self, *, seed: int=None, options: dict=None):
        super().reset(seed=seed)

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
                if self.use_vel:
                    state = np.concatenate((state, node["state"][2]))
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
        goal_cost_scaler = .1 if self.iter < len(self.guiding_path) else 1.
        e = eval_state - self.target_state
        self.reward = -goal_cost_scaler * (e.T @ e)
        
        # Distance to guiding path
        if self.guiding and self.iter < len(self.guiding_path):
            guiding_step = self.guiding_path[self.iter]
            e = eval_state - guiding_step
            self.reward -= e.T @ e

        truncated = self.iter >= self.max_steps
        terminated = truncated
        info = {}

        return self.state, self.reward, terminated, truncated, info

    def render(self, mode: str="") -> np.ndarray:

        print("Iter: ", self.iter, "Reward: ", self.reward)

        img = self.sim.renderImg()

        return img
