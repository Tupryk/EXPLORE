import os
import cma
import time
import psutil
import pickle
import hnswlib
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from explore.env.mujoco_sim import MjSim
import mujoco

def signum(q1, q2):
    if np.inner(q1, q2)>=0:
        return 1
    else:
        return -1

class MultiSearchNode:
    def __init__(self,
                 parent: int,
                 delta_q: np.ndarray,
                 state: tuple,
                 q_sequence: np.ndarray,
                 path: list=None,
                 explore_node: bool=False,
                 target_config_idx: int=-1):
        self.parent = parent
        self.delta_q = delta_q  # TODO: Remove this
        self.state = state
        self.path = path  # Motion
        self.explore_node = explore_node
        self.target_config_idx = target_config_idx
        self.q_sequence = q_sequence
        self.failed_expansion_count = 0

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

        self.config_count = self.configs.shape[0]

        self.max_nodes_per_tree = int(cfg.max_nodes_per_tree)
        self.v_0 = 0
        self.v_1 = 0
        self.v_2 = 0

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
        
        self.horizon = cfg.horizon
        self.sample_count = cfg.sample_count
        self.cost_method = cfg.cost_method # "se" for squared error, "max" for max error, sefo for se with derivates
        self.warm_start = cfg.warm_start
        self.sampling_strategy = cfg.sampling_strategy
        self.q_mask = np.array(cfg.q_mask)
        if not self.q_mask.shape[0]:
            self.q_mask = np.ones_like(self.configs[0])
            
        assert (self.warm_start and self.horizon == 1) or not self.warm_start
        
        self.sample_uniform_prob = cfg.sample_uniform_prob
        if self.sample_uniform_prob:
            self.mins_uniform_sample = np.array(cfg.mins_uniform_sample)
            self.maxs_uniform_sample = np.array(cfg.maxs_uniform_sample)

        # Does not always provide faster execution! Depends on weird factors like tau_sim
        self.threading = cfg.threading
        # Five Workers and ten simulators seem to be optimal
        self.max_workers = 5
        self.sim_count = 10 if self.threading else 1
        self.verbose = cfg.verbose
        if self.threading:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.start_ids = cfg.start_idx
        self.end_ids = cfg.end_idx
        if not isinstance(self.start_ids, ListConfig):
            if self.start_ids == -1:
                self.start_ids = list(range(self.config_count))
            else:
                self.start_ids = [self.start_ids]
        if not isinstance(self.end_ids, ListConfig):
            if self.end_ids == -1:
                self.end_ids = []
            else:
                self.end_ids = [self.end_ids]
        self.end_ids = np.array(self.end_ids)
        
        self.disable_node_max_strikes = cfg.disable_node_max_strikes
        self.n_best_actions = cfg.n_best_actions
        self.knnK = cfg.knnK
        self.regularization_weight = cfg.regularization_weight

        self.sim = [
            MjSim(
                self.mujoco_xml, self.tau_sim, view=False, verbose=0,
                interpolate=self.interpolate_actions,
                joints_are_same_as_ctrl=self.joints_are_same_as_ctrl,
                use_spline_ref=cfg.sim.use_spline_ref) for _ in range(self.sim_count)
        ]
        assert (not self.threading) or (self.sample_count % len(self.sim) == 0)
        
        self.ctrl_dim = self.sim[0].data.ctrl.shape[0]
        self.ctrl_ranges = self.sim[0].model.actuator_ctrlrange
        self.state_dim = self.sim[0].data.qpos.shape[0]
        
        if self.sampling_strategy == "rs":
            self.action_sampler = lambda o, t: self.random_sample_ctrls(o, t)
        elif self.sampling_strategy == "cma":
            self.action_sampler = lambda o, t: self.cma_sample_ctrls(o, t)
        elif self.sampling_strategy == "cem":
            self.cem_steps = cfg.cem_steps
            self.action_sampler = lambda o, t: self.cem_sample_ctrls(o, t)
        else:
            raise Exception(f"Sampling strategy '{self.sampling_strategy}' not implemented yet!")
            
        if self.verbose:
            print(f"Starting search across {self.config_count} configs!")

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
            
            self.sim[0].pushConfig(self.configs[i], self.configs_ctrl[i])
            state = self.sim[0].getState()
        
            root = MultiSearchNode(-1, np.zeros_like(self.configs_ctrl[0]), state, 0.)
            trees.append([root])
            
            self.trees_closest_nodes_idxs.append(np.full((self.config_count, self.knnK), -1))
            self.trees_closest_nodes_costs.append(np.full((self.config_count, self.knnK), np.nan))
            
            for ci in range(self.config_count):
                cost = self.compute_cost(self.configs[i], self.configs[ci])
                self.trees_closest_nodes_costs[i][ci, 0] = cost
                self.trees_closest_nodes_idxs[i][ci, 0] = 0

        return trees
    
    def gauss_sample_ctrl(self, parent_node: MultiSearchNode, sample_count: int=1,
                          std_perc: float=0.1, mean: np.ndarray=None) -> np.ndarray:

        if mean is None:
            mean = parent_node.delta_q if self.warm_start else np.zeros_like(parent_node.delta_q)
        if isinstance(self.stepsize, np.ndarray) or self.stepsize > 0.:
            std_devs = self.stepsize
        else:
            std_devs = np.abs(self.ctrl_ranges[:, 0] - self.ctrl_ranges[:, 1]) * std_perc
        noise = np.random.randn(sample_count * self.horizon * self.ctrl_dim)
        noise = noise.reshape(sample_count, self.horizon, self.ctrl_dim)
        delta = noise * std_devs + mean
        
        sample = delta + parent_node.state[3]
        sample = np.clip(sample, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])
        return sample
            
    def random_sample_ctrls(
            self,
            parent_node: MultiSearchNode,
            target: np.ndarray
        ) -> list[tuple[float, np.ndarray, np.ndarray]]:

        sampled_ctrls = self.gauss_sample_ctrl(parent_node, self.sample_count)

        results = self.eval_multiple_ctrls(sampled_ctrls, parent_node.state, target)
        
        if self.n_best_actions != -1:
            best_results = sorted(results, key=lambda x: x[0])[:self.n_best_actions]
        else:
            best_results = results

        return best_results
    
    def cma_sample_ctrls(
            self,
            parent_node: MultiSearchNode,
            target: np.ndarray
        ) -> list[tuple[float, np.ndarray, np.ndarray]]:

        pop_size = 200
        initial_guess = np.full(self.horizon, parent_node.state[3])

        es = cma.CMAEvolutionStrategy(
            initial_guess, 0.5,
            {
                "popsize": pop_size,
                "maxfevals": self.sample_count,
                "verbose": -1
            },
        )

        all_results = []

        while not es.stop():
            candidates = es.ask()

            results = self.eval_multiple_ctrls(
                candidates.reshape(-1, self.horizon, self.ctrl_dim), parent_node.state, target)
            fitness_values = [r[0] for r in results]

            all_results.extend(results)

            es.tell(candidates, fitness_values)

            if self.verbose > 3:
                es.disp()

        if self.n_best_actions != -1:
            best_results = sorted(all_results, key=lambda x: x[0])[:self.n_best_actions]
        else:
            best_results = all_results
        
        return best_results
    
    def cem_sample_ctrls(
            self,
            parent_node: MultiSearchNode,
            target: np.ndarray
        ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        
        q_offset = np.tile(parent_node.state[3], self.horizon).reshape(self.horizon, self.ctrl_dim)
        best_result = None
        mean = np.zeros_like(parent_node.delta_q)
        for _ in range(self.cem_steps):
            
            sampled_ctrls = self.gauss_sample_ctrl(parent_node, self.sample_count, mean=mean)
            if best_result is not None:
                sampled_ctrls[0] = best_result[2]

            results = self.eval_multiple_ctrls(sampled_ctrls, parent_node.state, target)
            best_result = min(results)
            mean = best_result[2] - q_offset

        return [best_result]
    
    def eval_multiple_ctrls_seq(self, ctrls: np.ndarray, origin: tuple,
                                target: tuple, sim_idx: int=0) -> list[tuple[float, np.ndarray, np.ndarray]]:
        results = []
        for ctrl in ctrls:
            res = self.eval_ctrl(ctrl, origin, target, sim_idx=sim_idx)
            results.append(res)
        return results
    
    def eval_multiple_ctrls(self, ctrls: np.ndarray, origin: tuple,
                            target: tuple) -> list[tuple[float, np.ndarray, np.ndarray]]:
        results = []
        
        if self.threading:
            sim_count = len(self.sim)
            sample_batch_count = len(ctrls) // sim_count
        
            futures = [
                self.executor.submit(
                    self.eval_multiple_ctrls_seq,
                    ctrls[
                        sim_idx * sample_batch_count
                        :
                        (sim_idx + 1) * sample_batch_count
                    ],
                    origin, target,
                    sim_idx
                )
                for sim_idx in range(sim_count)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        else:
            self.eval_multiple_ctrls_seq(ctrls, origin, target)
            
        
        if self.verbose > 3:
            results_plot = []
            min_results = results[0][0]
            results_plot.append(min_results)
            for res in results:
                new_res = res[0]
                if results_plot[-1] > new_res:
                    results_plot.append(new_res)
                else:
                    results_plot.append(results_plot[-1])
            
            plt.plot(results_plot)
            plt.show()
        
        return results
    
    def compute_cost(self, state1: np.ndarray, state2: np.ndarray, vel_state2: np.ndarray = None) -> float:
        e = (state1 - state2) #* self.q_mask
        if self.cost_method == "max":
            cost = np.abs(e).max()
        elif self.cost_method == "sefo" and vel_state2 is not None:

            e_linear = e[:-4]
            v_linear = vel_state2[:-3]
            

            delta_theta = np.zeros(3)
            mujoco.mju_subQuat(delta_theta, state2[-4:], state1[-4:])
            
            E_total = np.concatenate([e_linear, delta_theta])

            E_total *= self.q_mask

            V_total = np.concatenate([v_linear, vel_state2[-3:]]) 
            V_total *= self.q_mask
            
            v_max = 1
            alpha = (1 - 1e-3) / v_max
            
            dist = np.linalg.norm(E_total)
            alignment = np.inner(V_total, E_total)
            
            cost = (dist - alpha * alignment)**2
        elif self.cost_method == "se":
            e_linear = e[:-4]
            
            #print("SIGN:", np.inner(state2[-4:], state1[-4:]))

            e_rotation = np.array([0,0,0,0])# state1[-4:] - signum(state1[-4:], state2[-4:]) * state2[-4:]

            E_total = np.concatenate([e_linear, e_rotation])
            
            E_total *= self.q_mask
            
            #print("norms:", np.linalg.norm(E_total[:9])**2, "quat:", np.linalg.norm(E_total[-3])**2)
            cost = np.linalg.norm(E_total)**2
        else:
            raise Exception("Cost method '{}' not implemented yet!")
        return cost
    
    def eval_ctrl(self, ctrl: np.ndarray, origin: tuple,
                  target: np.ndarray, sim_idx: int=0
                  ) -> tuple[float, np.ndarray, np.ndarray]:
        
        self.sim[sim_idx].setState(*origin)
        
        for c in ctrl:
            self.sim[sim_idx].step(self.tau_action, c)
        
        state = self.sim[sim_idx].getState()
        
        v0 = np.linalg.norm(state[2][:3])
        v1 = np.linalg.norm(state[2][3:6])
        v2 = np.linalg.norm(state[2][6:9])

        if v0 > self.v_0:
            self.v_0 = v0
        if v1 > self.v_1:
            self.v_1 = v1
        if v2 > self.v_2:
            self.v_2 = v2

        cost2target = self.compute_cost(target, state[1], state[2])
        
        reg_e = state[3] - origin[3].T
        cost2target += self.regularization_weight * (reg_e.T @ reg_e)
        
        return cost2target, state, ctrl
    
    def store_tree(self, idx: int, folder_path: str, trees: list):
        dict_tree = []
        for node in trees[idx]:
            new_node = {
                "parent": node.parent,
                "delta_q": node.delta_q,
                "q_sequence": node.q_sequence,
                "state": node.state,
                "path": node.path,
                "explore_node": node.explore_node,
                "target_config_idx": node.target_config_idx
            }
            dict_tree.append(new_node)
            
        data_path = os.path.join(folder_path, f"tree_{idx}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(dict_tree, f)
        
        stats_path = os.path.join(folder_path, f"tree_stats_{idx}.txt")
        with open(stats_path, "w") as f:
            costs = self.trees_closest_nodes_costs[idx][:, 0]

            if 0 <= idx < len(costs):
                costs = costs[np.arange(len(costs)) != idx]

            mean_cost = costs.mean()
            min_cost = costs.min()

            f.write(f"{mean_cost}, {min_cost}")
            if len(self.end_ids) == 1:
                f.write(f", {self.trees_closest_nodes_costs[idx][self.end_ids[0], 0]}\n")
            else:
                f.write("\n")

    def sample_state(self, start_idx: int = -1) -> tuple[np.ndarray, int]:
        """
        Sampling rules:
        1. Prefer values < self.target_min_dist and > self.min_cost
        2. Ignore start_idx if it is not -1
        3. If none satisfy (1), pick any value > self.min_cost
        4. If none satisfy (1) or (3), pick any random value (except start_idx if possible)
        """

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

        preferred_mask = below_target # & above_min
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
        
        folder_path = os.path.join(self.output_dir, "trees")
        os.makedirs(folder_path, exist_ok=True)
        
        [self.store_tree(i, folder_path, self.trees) for i in range(self.config_count)]
        
        start_time = time.time()
        
        for i, start_idx in enumerate(self.start_ids):

            if self.sample_uniform_prob:
                max_elems = self.n_best_actions * self.max_nodes_per_tree
                kNN_tree = hnswlib.Index(space="l2", dim=self.state_dim)
                kNN_tree.init_index(max_elements=max_elems, ef_construction=200, M=16)
                kNN_tree.add_items(self.configs[start_idx], ids=[0])
                kNN_tree_size = 1

            if self.verbose > 0:
                pbar = trange(self.max_nodes_per_tree, desc=f"Tree {i+1}/{len(self.start_ids)}", unit="nodes")
            else:
                pbar = range(self.max_nodes_per_tree)
            
            for _ in pbar:

                # Sample random sim state
                exploring = not (np.random.uniform() < self.target_prob) or not len(self.end_ids)
                
                if not exploring:
                    target_config_idx = np.random.choice(self.end_ids)
                    sim_sample = self.configs[target_config_idx]
                
                else:
                    sim_sample, target_config_idx = self.sample_state(start_idx)
                
                # Pick closest node
                if self.sample_uniform_prob and exploring and (np.random.uniform() <  self.sample_uniform_prob):

                    target_config_idx = -1
                    
                    sim_sample = np.random.uniform(low=self.mins_uniform_sample, high=self.maxs_uniform_sample)

                    k = min(self.knnK, kNN_tree_size)
                    node_id, _ = kNN_tree.knn_query(sim_sample * self.q_mask, k=k)
                    node_id = np.random.choice(node_id[0])

                else:
                    node_ids = self.trees_closest_nodes_idxs[start_idx][target_config_idx]
                    valid_ids = node_ids[node_ids != -1]
                    if len(valid_ids):
                        node_id = np.random.choice(valid_ids)
                    else:
                        node_id = 0
                    assert node_id != -1

                node: MultiSearchNode = self.trees[start_idx][node_id]

                # Expand node
                expanded = False
                best_expansions = self.action_sampler(node, sim_sample)
                for best_node_cost, best_state, best_q in best_expansions:
                    
                    store_node = False
                    for ci in range(self.config_count):
                        new_cost = self.compute_cost(best_state[1], self.configs[ci])
                        for k in range(self.knnK):
                            stored_cost = self.trees_closest_nodes_costs[start_idx][ci][k]
                            if new_cost < self.target_min_dist and stored_cost >= new_cost:
                                expanded = True
                                store_node = True
                                
                                # Shift values
                                self.trees_closest_nodes_costs[start_idx][ci, k+1:] = self.trees_closest_nodes_costs[start_idx][ci, k:-1]
                                self.trees_closest_nodes_idxs[start_idx][ci, k+1:] = self.trees_closest_nodes_idxs[start_idx][ci, k:-1]
                                
                                self.trees_closest_nodes_costs[start_idx][ci][k] = new_cost
                                self.trees_closest_nodes_idxs[start_idx][ci][k] = len(self.trees[start_idx]) - 1
                                break
                    
                    if store_node:
                        delta_q = (best_q - node.state[3]).copy()
                        best_node = MultiSearchNode(
                            node_id, delta_q, best_state,
                            explore_node=exploring,
                            target_config_idx=target_config_idx,
                            q_sequence=best_q)
                        self.trees[start_idx].append(best_node)
                        
                        if self.sample_uniform_prob:
                            kNN_tree.add_items(best_state[1] * self.q_mask, ids=[kNN_tree_size])
                            kNN_tree_size += 1

                if not expanded:

                    self.trees[start_idx][node_id].failed_expansion_count += 1
                    
                    if (
                        self.disable_node_max_strikes != -1 and
                        self.trees[start_idx][node_id].failed_expansion_count >= self.disable_node_max_strikes
                        ):

                        if self.verbose > 2:
                            print(f"Node #{node_id} attempt at expansion failed {self.disable_node_max_strikes} time(s)! Killing it >:(")
                        
                        for ci in range(self.config_count):
                            for k in range(self.knnK):
                                
                                if self.trees_closest_nodes_idxs[start_idx][ci, k] == node_id:

                                    self.trees_closest_nodes_costs[start_idx][ci, k:-2] = self.trees_closest_nodes_costs[start_idx][ci, k+1:-1]
                                    self.trees_closest_nodes_idxs[start_idx][ci, k:-2] = self.trees_closest_nodes_idxs[start_idx][ci, k+1:-1]
                                    
                                    self.trees_closest_nodes_costs[start_idx][ci, -1] = np.nan
                                    self.trees_closest_nodes_idxs[start_idx][ci, -1] = -1

                                    break
                                
                                elif self.trees_closest_nodes_idxs[start_idx][ci, k] == -1:
                                    break
                            
                            if np.isnan(self.trees_closest_nodes_costs[start_idx][ci, 0]):
                                self.trees_closest_nodes_costs[start_idx][ci, 0] = self.compute_cost(self.trees[start_idx][0].state[1], self.configs[ci])
                                self.trees_closest_nodes_idxs[start_idx][ci, 0] = 0

                    elif self.verbose > 2:
                        print(f"Node #{node_id} did not expand the tree and got a strike (total strikes for this node: {self.trees[start_idx][node_id].failed_expansion_count}/{self.disable_node_max_strikes}).")
                    
                if self.verbose > 1:
                    costs = self.trees_closest_nodes_costs[start_idx][:, 0]

                    if 0 <= start_idx < len(costs):
                        costs = costs[np.arange(len(costs)) != start_idx]

                    mean_cost = costs.mean()
                    min_cost = costs.min()

                    print(f"Mean Cost: {mean_cost} | Lowest Cost: {min_cost}", end="")
                    if len(self.end_ids) == 1:
                        print(f" | Cost to end_idx {self.trees_closest_nodes_costs[start_idx][self.end_ids[0], 0]}")
                    else:
                        print()
                print(f"Mean Cost: {mean_cost} | Lowest Cost: {min_cost} vs: {self.v_0} {self.v_1} {self.v_2}",end="")
                if len(self.end_ids)==1:
                    print(f" | Cost to end_idx {self.trees_closest_nodes_costs[start_idx][self.end_ids[0], 0]}")
                else:
                    print()

            # Store information when appropriate
            if self.verbose > 3:
                print(f"Storing tree {start_idx}")
            
            self.store_tree(start_idx, folder_path, self.trees)
            
            # Free memory
            self.trees = self.init_trees()

            if self.verbose > 6:
                process = psutil.Process(os.getpid())
                print(f"RSS (resident memory): {process.memory_info().rss / 1024**2:.2f} MB")
                print(f"VMS (virtual memory): {process.memory_info().vms / 1024**2:.2f} MB")
            
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.verbose > 1:
            print(f"Total time taken: {total_time:.2f} seconds")

        print("v0:", self.v_0, "v1:", self.v_1, "v2:", self.v_2)
        time_data_path = os.path.join(self.output_dir, f"time_taken.txt")
        with open(time_data_path, "w") as f:
            f.write(f"{total_time}\n")
