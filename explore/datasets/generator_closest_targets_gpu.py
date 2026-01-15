import os
import jax
import time
import mujoco
import psutil
import pickle
import numpy as np
from mujoco import mjx
import jax.numpy as jnp
from tqdm import trange
from hydrax.algs import MPPI
from omegaconf import DictConfig, ListConfig

from explore.env.tasks import StaGE_task


class MultiSearchNode:
    def __init__(self,
                 parent: int,
                 state: tuple,
                 q_sequence: np.ndarray,
                 explore_node: bool=False,
                 target_config_idx: int=-1):
        self.parent = parent
        self.state = state
        self.explore_node = explore_node
        self.target_config_idx = target_config_idx
        self.q_sequence = q_sequence

class Search:

    def __init__(self, configs: np.ndarray, configs_ctrl: np.ndarray, cfg: DictConfig):
        
        assert len(configs) == len(configs_ctrl)
        self.max_configs = cfg.max_configs

        if self.max_configs != -1 and len(configs) > self.max_configs:
            self.configs = configs[:self.max_configs]
            self.configs_ctrl = configs_ctrl[:self.max_configs]
        else:
            self.configs = configs
            self.configs_ctrl = configs_ctrl

        self.max_nodes = int(cfg.max_nodes)
        self.stepsize = cfg.stepsize
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize)
        self.target_prob = cfg.target_prob
        self.target_min_dist = cfg.target_min_dist
        
        self.min_cost = cfg.min_cost
        self.output_dir = cfg.output_dir
        
        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        self.mujoco_xml = cfg.sim.mujoco_xml
        
        self.cost_max_method = cfg.cost_max_method
        self.sample_uniform = cfg.sample_uniform
        self.q_mask = np.array(cfg.q_mask)
        if not self.q_mask.shape[0]:
            self.q_mask = np.ones_like(self.configs[0])
            
        self.verbose = cfg.verbose
        
        self.start_idx = cfg.start_idx
        self.end_idx = cfg.end_idx
        
        self.knnK = cfg.knnK

        self.task = StaGE_task(self.mujoco_xml, self.tau_sim, self.q_mask, self.cost_max_method)
        self.mj_model = self.task.mj_model
        self.mj_data = mujoco.MjData(self.mj_model)
        # self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer = None

        self.mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
        self.mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

        self.controller = MPPI(
            self.task,
            num_samples=128,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=4,
            plan_horizon=0.6,
            spline_type="zero",
            num_knots=4,
        )

        self.ctrl_dim = self.mj_data.ctrl.shape[0]
        self.ctrl_ranges = self.mj_model.actuator_ctrlrange
        self.state_dim = self.mj_data.qpos.shape[0]
        self.config_count = self.configs.shape[0]
            
        if self.verbose:
            print(f"Starting search across {self.config_count} configs!")
        
    def jit_simulator(self):
        # Report the planning horizon in seconds for debugging
        print(
            f"Planning with {self.controller.ctrl_steps} steps "
            f"over a {self.controller.plan_horizon} second horizon "
            f"with {self.controller.num_knots} knots."
        )

        # Figure out how many sim steps to run before replanning
        frequency = 10
        replan_period = 1.0 / frequency
        self.sim_steps_per_replan = int(replan_period / self.mj_model.opt.timestep)
        self.sim_steps_per_replan = max(self.sim_steps_per_replan, 1)
        step_dt = self.sim_steps_per_replan * self.mj_model.opt.timestep
        actual_frequency = 1.0 / step_dt
        print(
            f"Planning at {actual_frequency} Hz, "
            f"simulating at {1.0 / self.mj_model.opt.timestep} Hz"
        )

        # Initialize the controller
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        self.policy_params = self.controller.init_params(initial_knots=None)
        self.jit_optimize = jax.jit(self.controller.optimize)
        self.jit_interp_func = jax.jit(self.controller.interp_func)

        # Warm-up the controller
        print("Jitting the controller...")
        st = time.time()
        self.policy_params, rollouts = self.jit_optimize(self.mjx_data, self.policy_params)
        self.policy_params, rollouts = self.jit_optimize(self.mjx_data, self.policy_params)

        tq = jnp.arange(0, self.sim_steps_per_replan) * self.mj_model.opt.timestep
        tk = self.policy_params.tk
        knots = self.policy_params.mean[None, ...]
        _ = self.jit_interp_func(tq, tk, knots)
        _ = self.jit_interp_func(tq, tk, knots)
        print(f"Time to jit: {time.time() - st:.3f} seconds")

    def extend_node(self):

        traj = []

        # for _ in range(25):
        # Set the start state for the controller
        self.mjx_data = self.mjx_data.replace(
            qpos=jnp.array(self.mj_data.qpos),
            qvel=jnp.array(self.mj_data.qvel),
            time=self.mj_data.time
        )

        # Do a replanning step
        self.policy_params, rollouts = self.jit_optimize(self.mjx_data, self.policy_params)

        # query the control spline at the sim frequency
        # (we assume the sim freq is the same as the low-level ctrl freq)
        sim_dt = self.mj_model.opt.timestep
        t_curr = self.mj_data.time

        tq = jnp.arange(0, self.sim_steps_per_replan) * sim_dt + t_curr
        tk = self.policy_params.tk
        knots = self.policy_params.mean[None, ...]
        us = np.asarray(self.jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

        # simulate the system between spline replanning steps
        for i in range(self.sim_steps_per_replan):
            
            new_ctrl = np.array(us[i])
            self.mj_data.ctrl[:] = new_ctrl
            
            mujoco.mj_step(self.mj_model, self.mj_data)
            traj.append(new_ctrl)

            if self.viewer is not None:
                self.viewer.sync()

        end_state = (
            self.mj_data.time,
            np.copy(self.mj_data.qpos),
            np.copy(self.mj_data.qvel),
            np.copy(self.mj_data.ctrl)
        )

        return end_state, traj

    def init_trees(self) -> tuple:
        
        # TODO: It does not make sense to have a list with all trees if we only use one at a time

        trees: list[list[MultiSearchNode]] = []
        self.trees_closest_nodes_idxs = []
        self.trees_closest_nodes_costs = []

        print("Initializing trees...")
        
        if self.verbose > 1:
            pbar = trange(self.config_count, desc="Init trees", unit="tree")
        else:
            pbar = range(self.config_count)
            
        for i in pbar:
            
            state = (
                0.0,
                self.configs[i],
                np.zeros_like(self.mj_data.qvel),
                self.configs_ctrl[i]
            )
        
            root = MultiSearchNode(-1, state, [])
            trees.append([root])
            
            self.trees_closest_nodes_idxs.append(np.full((self.config_count, self.knnK), -1))
            self.trees_closest_nodes_costs.append(np.full((self.config_count, self.knnK), np.nan))
            
            for ci in range(self.config_count):
                cost = self.compute_cost(self.configs[i], self.configs[ci])
                self.trees_closest_nodes_costs[i][ci, 0] = cost
                if cost <= self.target_min_dist:
                    self.trees_closest_nodes_idxs[i][ci, 0] = 0
                else:
                    self.trees_closest_nodes_idxs[i][ci, 0] = -1

        return trees
    
    def compute_cost(self, state1: np.ndarray, state2: np.ndarray) -> float:
        e = (state1 - state2) * self.q_mask
        if self.cost_max_method:
            cost = np.abs(e).max()
        else:
            cost = e.T @ e
        return cost
    
    def store_tree(self, idx: int, folder_path: str, trees: list[list[MultiSearchNode]]):
        dict_tree = []
        for node in trees[idx]:
            new_node = {
                "parent": node.parent,
                "q_sequence": node.q_sequence,
                "state": node.state,
                "explore_node": node.explore_node,
                "target_config_idx": node.target_config_idx
            }
            dict_tree.append(new_node)
            
        data_path = os.path.join(folder_path, f"tree_{idx}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(dict_tree, f)
        
        stats_path = os.path.join(self.output_dir, "stats", f"tree_stats_{idx}.txt")
        with open(stats_path, "w") as f:
            costs = self.trees_closest_nodes_costs[idx][:, 0]

            if 0 <= idx < len(costs):
                costs = costs[np.arange(len(costs)) != idx]

            mean_cost = costs.mean()
            min_cost = costs.min()

            f.write(f"{mean_cost}, {min_cost}")
            if self.end_idx != -1:
                f.write(f", {self.trees_closest_nodes_costs[idx][self.end_idx, 0]}\n")
            else:
                f.write("\n")

    def sample_state(self, start_idx: int = -1) -> tuple[np.ndarray, int]:
    
        costs = self.trees_closest_nodes_costs[start_idx][:, 0]
        num_states = len(costs)

        all_indices = np.arange(num_states)

        if start_idx != -1:
            valid_indices = all_indices[all_indices != start_idx]
        else:
            valid_indices = all_indices

        # Masks
        below_target = costs < self.target_min_dist
        above_min = costs > self.min_cost

        preferred_mask = below_target & above_min
        preferred_indices = valid_indices[preferred_mask[valid_indices]]

        # Print percentages (debug/info)
        if self.verbose > 1:
            valid_costs = costs[valid_indices]
            pct_below_target = np.mean(valid_costs < self.target_min_dist) * 100
            pct_below_min = np.mean(valid_costs < self.min_cost) * 100
            print(
                f"Below target_min_dist: {pct_below_target:.2f}% | "
                f"Below min_cost: {pct_below_min:.2f}%"
            )

        if len(preferred_indices) > 0:
            target_config_idx = np.random.choice(preferred_indices)

        else:
            fallback_indices = valid_indices[above_min[valid_indices]]

            if len(fallback_indices) > 0:
                target_config_idx = np.random.choice(fallback_indices)
            else:
                # Absolute fallback: pick anything except start_idx if possible
                if len(valid_indices) == 0:
                    target_config_idx = start_idx
                else:
                    target_config_idx = np.random.choice(valid_indices)

        sim_sample = self.configs[target_config_idx]
        return sim_sample, target_config_idx

    def run(self):
        
        self.trees = self.init_trees()

        self.jit_simulator()
        folder_path = os.path.join(self.output_dir, "trees")
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "stats"), exist_ok=True)
        
        [self.store_tree(i, folder_path, self.trees) for i in range(self.config_count)]
        
        if self.verbose > 0:
            pbar = trange(self.max_nodes, desc="Physics RRT Search", unit="nodes")
        else:
            pbar = range(self.max_nodes)
            
        nodes_per_tree = self.max_nodes // self.config_count
        
        start_time = time.time()
        
        assert (
            (self.start_idx != -1 and self.knnK <= self.max_nodes * 0.25) or
            (self.start_idx == -1 and self.knnK <= nodes_per_tree * 0.25)
        )
        
        for i in pbar:
            
            if self.start_idx == -1:
                start_idx = i // nodes_per_tree
                if start_idx >= self.config_count:  # TODO: Write a better fix
                    start_idx = self.config_count-1
            else:
                start_idx = self.start_idx

            # Sample random sim state
            exploring = not (np.random.uniform() < self.target_prob) or self.end_idx == -1
            
            if not exploring and self.end_idx != -1:
                target_config_idx = self.end_idx
                sim_sample = self.configs[target_config_idx]
            else:
                sim_sample, target_config_idx = self.sample_state(start_idx)
            
            # Pick closest node
            node_ids = self.trees_closest_nodes_idxs[start_idx][target_config_idx]
            valid_ids = node_ids[node_ids != -1]
            if len(valid_ids):
                node_id = np.random.choice(valid_ids)
            else:
                node_id = 0
            assert node_id != -1
            node: MultiSearchNode = self.trees[start_idx][node_id]

            self.mj_data.time = node.state[0]
            self.mj_data.qpos[:] = node.state[1]
            self.mj_data.qvel[:] = node.state[2]
            self.mj_data.ctrl[:] = node.state[3]

            self.controller.task.target_state = sim_sample

            end_state, traj = self.extend_node()

            new_node = MultiSearchNode(
                node_id,
                end_state,
                q_sequence=traj,
                target_config_idx=target_config_idx,
                explore_node=exploring
            )
            
            self.trees[start_idx].append(new_node)
            
            for ci in range(self.config_count):
                new_cost = self.compute_cost(end_state[1], self.configs[ci])
                for k in range(self.knnK):
                    stored_cost = self.trees_closest_nodes_costs[start_idx][ci][k]
                    if stored_cost >= new_cost:
                        
                        # Shift values
                        self.trees_closest_nodes_costs[start_idx][ci, k+1:] = self.trees_closest_nodes_costs[start_idx][ci, k:-1]
                        self.trees_closest_nodes_idxs[start_idx][ci, k+1:] = self.trees_closest_nodes_idxs[start_idx][ci, k:-1]
                        
                        self.trees_closest_nodes_costs[start_idx][ci][k] = new_cost
                        self.trees_closest_nodes_idxs[start_idx][ci][k] = len(self.trees[start_idx]) - 1
                        break
                
            if self.verbose > 1:
                costs = self.trees_closest_nodes_costs[start_idx][:, 0]

                if 0 <= start_idx < len(costs):
                    costs = costs[np.arange(len(costs)) != start_idx]

                mean_cost = costs.mean()
                min_cost = costs.min()

                print(f"Mean Cost: {mean_cost} | Lowest Cost: {min_cost}", end="")
                if self.end_idx != -1:
                    print(f" | Cost to end_idx {self.trees_closest_nodes_costs[start_idx][self.end_idx, 0]}")
                else:
                    print()

            # Store information when appropriate
            if (((self.start_idx == -1) and (i % nodes_per_tree == nodes_per_tree-1)) or
                ((self.start_idx != -1) and (i == self.max_nodes-1))):
                if self.verbose > 3:
                    print(f"Storing tree {start_idx} at i {i}")
                
                self.store_tree(start_idx, folder_path, self.trees)
                
                # Free memory
                self.trees = self.init_trees()

            if self.verbose > 2 and (i+1) % 1000 == 0:
                process = psutil.Process(os.getpid())
                print(f"RSS (resident memory): {process.memory_info().rss / 1024**2:.2f} MB")
                print(f"VMS (virtual memory): {process.memory_info().vms / 1024**2:.2f} MB")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.verbose > 1:
            print(f"Total time taken: {total_time:.2f} seconds")

        time_data_path = os.path.join(self.output_dir, f"time_taken.txt")
        with open(time_data_path, "w") as f:
            f.write(f"{total_time}\n")
