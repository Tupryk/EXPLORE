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
from explore.utils.utils import randint_excluding


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

class Search:

    def __init__(self, configs: np.ndarray, configs_ctrl: np.ndarray, cfg: DictConfig):
        
        self.configs = configs
        self.configs_ctrl = configs_ctrl

        self.max_nodes = int(cfg.max_nodes)
        self.stepsize = cfg.stepsize
        if isinstance(self.stepsize, ListConfig):
            self.stepsize = np.array(self.stepsize)
        self.target_prob = cfg.target_prob
        self.k_nearest_targets = cfg.k_nearest_targets
        
        self.min_cost = cfg.min_cost
        self.output_dir = cfg.output_dir
        
        self.tau_sim = cfg.sim.tau_sim
        self.tau_action = cfg.sim.tau_action
        self.interpolate_actions = cfg.sim.interpolate_actions
        self.joints_are_same_as_ctrl = cfg.sim.joints_are_same_as_ctrl
        self.mujoco_xml = cfg.sim.mujoco_xml
        
        self.horizon = cfg.horizon
        self.sample_count = cfg.sample_count
        self.cost_max_method = cfg.cost_max_method
        self.sample_uniform = cfg.sample_uniform
        self.warm_start = cfg.warm_start
        self.sampling_strategy = cfg.sampling_strategy
        self.q_mask = np.array(cfg.q_mask)
        if not self.q_mask.shape[0]:
            self.q_mask = np.ones_like(self.configs[0])
            
        assert (self.warm_start and self.horizon == 1) or not self.warm_start
        assert (cfg.sim.use_spline_ref and self.horizon == 1) or not cfg.sim.use_spline_ref
        
        # Does not always provide faster execution! Depends on weird factors like tau_sim
        # Does not work for the bi-tree for now
        self.threading = cfg.threading
        # Five Workers and ten simulators seem to be optimal
        self.max_workers = 5
        self.sim_count = 10 if self.threading else 1
        self.verbose = cfg.verbose
        
        self.start_idx = cfg.start_idx
        self.end_idx = cfg.end_idx
        self.bidirectional = cfg.bidirectional  # Performance seems very dependend on initial seed. Needs further investigation...
        self.knnK = cfg.knnK
        self.n_best_actions = cfg.n_best_actions
        self.regularization_weight = cfg.regularization_weight

        if self.bidirectional:
            self.bi_stepsize = cfg.bi_stepsize
            if isinstance(self.bi_stepsize, ListConfig):
                self.bi_stepsize = np.array(self.bi_stepsize)
            self.bi_target_prob = cfg.bi_target_prob
            self.bi_tree_tolerance = cfg.bi_tree_tolerance

        if self.bidirectional and self.end_idx == -1:
            print("It is recomended to have a specific target config when using bidirectional search.")
        assert not self.bidirectional or self.joints_are_same_as_ctrl
        
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
        self.config_count = self.configs.shape[0]
        
        if self.sampling_strategy == "rs":
            self.action_sampler = lambda o, t: self.random_sample_ctrls(o, t)
        elif self.sampling_strategy == "cma":
            self.action_sampler = lambda o, t: self.cma_sample_ctrls(o, t)
        elif self.sampling_strategy == "cem":
            self.cem_steps = cfg.cem_steps
            self.action_sampler = lambda o, t: self.cem_sample_ctrls(o, t)
        else:
            raise Exception(f"Sampling strategy '{self.sampling_strategy}' not implemented yet!")
        
        if self.bidirectional:
            if self.sampling_strategy == "rs":
                self.backward_action_sampler = lambda o, t: self.backward_random_sample_ctrls(o, t)
            else:
                raise Exception(f"Sampling strategy '{self.sampling_strategy}' not implemented for backward search yet!")
        
        self.kNN_targets = None
        if self.k_nearest_targets != -1:
            
            self.kNN_targets = hnswlib.Index(space='l2', dim=self.configs.shape[1])
            self.kNN_targets.init_index(max_elements=self.config_count, ef_construction=200, M=16)
            
            for i in range(self.config_count):
                new_state = self.configs[i]
                self.kNN_targets.add_items(new_state.astype(np.float32), ids=[i])
            
        if self.verbose:
            print(f"Starting search across {self.config_count} configs!")

    def init_trees(self) -> tuple:
        
        # TODO: It does not make sense to have a list with all trees if we only use one at a time

        trees: list[list[MultiSearchNode]] = []
        trees_kNNs: list[hnswlib.Index] = []
        trees_kNNs_sizes: list[int] = []

        if self.start_idx == -1:
            max_knn_tree_size = int(np.ceil(self.max_nodes / self.config_count) + 1)
        else:
            max_knn_tree_size = int(self.max_nodes + self.config_count)
        action_mutliplier = self.sample_count if self.n_best_actions == -1 else self.n_best_actions
        max_knn_tree_size *= action_mutliplier

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

            kNN_tree = hnswlib.Index(space='l2', dim=state[1].shape[0])
            kNN_tree.init_index(max_elements=max_knn_tree_size, ef_construction=200, M=16)
            kNN_tree.add_items(state[1].astype(np.float32), ids=[0])

            trees_kNNs.append(kNN_tree)
            trees_kNNs_sizes.append(1)
        
        return trees, trees_kNNs, trees_kNNs_sizes
    
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
        sample = np.clip(sample, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])  # Might be slow...
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
                sampled_ctrls = np.concatenate((best_result[2][None], sampled_ctrls), axis=0)

            results = self.eval_multiple_ctrls(sampled_ctrls, parent_node.state, target)
            best_result = sorted(results, key=lambda x: x[0])[0]  # TODO: make faster maybe
            mean = best_result[2] - q_offset

        return [best_result]
    
    def backward_random_sample_ctrls(self, to_node: MultiSearchNode, from_target: np.ndarray
                                     ) -> tuple[float, np.ndarray, np.ndarray]:
        
        # TODO: Sample distance to then sample noise on the plane defined by the distance vector as the normal
        # TODO: don't ignore velocities
        # Sample starting states
        from_to_vec  = from_target - to_node.state[1]
        from_to_vec_normed = from_to_vec / np.linalg.norm(from_to_vec)
        from_to_vec_scaled = from_to_vec_normed * self.bi_stepsize
        from_state = from_to_vec_scaled + to_node.state[1]
        
        noise = np.random.randn(self.sample_count * self.state_dim) * self.bi_stepsize * .5
        noise = noise.reshape(self.sample_count, self.state_dim)
        states = noise + from_state

        origin = [
            to_node.state[0] - self.tau_action,
            to_node.state[1].copy(),
            np.zeros_like(to_node.state[2]),
            to_node.state[3].copy()
        ]
        best_cost = 1e6

        if self.threading:
            sim_count = len(self.sim)
            sample_batch_count = self.sample_count // sim_count
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(sample_batch_count):
                    s = states[i]
                    origin[1] = s
                    origin[3] = s[:self.ctrl_dim]
                    futures = [
                        executor.submit(
                            self.eval_ctrl,
                            to_node.state[3],
                            origin, to_node.state[1],
                            sim_idx, True
                        )
                        for sim_idx in range(sim_count)
                    ]

                    for future in as_completed(futures):
                        cost, s = future.result()[0], states[futures.index(future)]
                        if cost < best_cost:
                            best_cost = cost
                            best_start_state = s
        else:
            for s in states:
                origin[1] = s
                origin[3] = s[:self.ctrl_dim]
                res = self.eval_ctrl(to_node.state[3], origin, to_node.state[1], max_method=True)
                if res[0] < best_cost:
                    best_cost = res[0]
                    best_start_state = s

        return best_cost, best_start_state
    
    def eval_multiple_ctrls(self, ctrls: np.ndarray, origin: tuple,
                            target: tuple) -> list[tuple[float, np.ndarray, np.ndarray]]:
        results = []
        
        if self.threading:
            sim_count = len(self.sim)
            sample_batch_count = len(ctrls) // sim_count
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(sample_batch_count):
                    futures = [
                        executor.submit(
                            self.eval_ctrl,
                            ctrls[i*sim_count+sim_idx],
                            origin, target,
                            sim_idx, self.cost_max_method
                        )
                        for sim_idx in range(sim_count)
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
        else:
            for ctrl in ctrls:
                res = self.eval_ctrl(ctrl, origin, target, max_method=self.cost_max_method)
                results.append(res)
        
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
    
    def eval_ctrl(self, ctrl: np.ndarray, origin: tuple,
                  target: np.ndarray, sim_idx: int=0, max_method: bool=False
                  ) -> tuple[float, np.ndarray, np.ndarray]:
        
        self.sim[sim_idx].setState(*origin)
        
        for c in ctrl:
            self.sim[sim_idx].step(self.tau_action, c)
        
        state = self.sim[sim_idx].getState()
        
        e = (target - state[1]) * self.q_mask
        if max_method:
            cost2target = np.abs(e).max()
        else:
            cost2target = e.T @ e
        
        reg_e = state[3] - origin[3].T
        cost2target += self.regularization_weight *  (reg_e.T @ reg_e)
        
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

    def sample_state(self, start_idx: int=-1) -> np.ndarray:
        if self.sample_uniform:
            # TODO: Sampling ranges are not the same for each scene and each dim in the qpos...
            target_config_idx = -1
            sim_sample = np.random.uniform(low=-1., high=1., size=self.state_dim)
        else:
            target_config_idx = randint_excluding(0, self.config_count, start_idx)  # TODO: exclude end_idx if end_idx != -1
            sim_sample = self.configs[target_config_idx]
        return sim_sample, target_config_idx

    def run(self) -> tuple[list[MultiSearchNode], float]:
        
        self.trees, self.trees_kNNs, self.trees_kNNs_sizes = self.init_trees()
        if self.bidirectional:
            self.bi_trees, self.bi_trees_kNNs, self.bi_trees_kNNs_sizes = self.init_trees()
        
        folder_path = os.path.join(self.output_dir, "trees")
        os.makedirs(folder_path, exist_ok=True)
        
        [self.store_tree(i, folder_path, self.trees) for i in range(self.config_count)]
        
        if self.bidirectional:
            bi_folder_path = os.path.join(self.output_dir, "bi_trees")
            os.makedirs(bi_folder_path, exist_ok=True)
            
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
            
            if self.k_nearest_targets == -1:
                
                if exploring or self.end_idx == -1:
                    sim_sample, target_config_idx = self.sample_state(start_idx)
                else:
                    target_config_idx = self.end_idx
                    sim_sample = self.configs[target_config_idx]
                
                # Pick closest node
                k = min(self.knnK, self.trees_kNNs_sizes[start_idx])
                node_idx, _ = self.trees_kNNs[start_idx].knn_query(sim_sample * self.q_mask, k=k)
                node_idx = np.random.choice(node_idx[0])
                node: MultiSearchNode = self.trees[start_idx][node_idx]
            
            else:
                
                tree_size = len(self.trees[start_idx])
                sample_size = 32 if tree_size > 32 else tree_size
                
                node_idx = np.random.randint(sample_size)
                node_idx = int(tree_size - sample_size + node_idx)
                
                node: MultiSearchNode = self.trees[start_idx][node_idx]
                node_state = node.state[1].astype(np.float32) * self.q_mask
                
                target_config_idx, _ = self.kNN_targets.knn_query(node_state, k=self.k_nearest_targets)
                target_config_idx = np.random.choice(target_config_idx[0])
                sim_sample = self.configs[target_config_idx]

            # Expand node
            best_expansions = self.action_sampler(node, sim_sample)
            for best_node_cost, best_state, best_q in best_expansions:
                delta_q = (best_q - node.state[3]).copy()
                best_node = MultiSearchNode(
                    node_idx, delta_q, best_state,
                    explore_node=exploring,
                    target_config_idx=target_config_idx,
                    q_sequence=best_q)
                
                self.trees[start_idx].append(best_node)
                knn_item = best_state[1].astype(np.float32) * self.q_mask
                self.trees_kNNs[start_idx].add_items(knn_item, ids=[self.trees_kNNs_sizes[start_idx]])
                self.trees_kNNs_sizes[start_idx] += 1

            # Try to expand bi_tree backwards
            if self.bidirectional:
                if self.end_idx == -1:
                    bi_tree_idxs = range(self.config_count)
                else:
                    bi_tree_idxs = [self.end_idx]
                
                for bi_tree_idx in bi_tree_idxs:
                    bi_exploring = not (np.random.uniform() < self.bi_target_prob)
                    if bi_exploring:
                        from_state, target_config_idx = self.sample_state()
                    else:
                        from_state = best_node.state[1]
                        target_config_idx = -1
                    node_idx, _ = self.bi_trees_kNNs[bi_tree_idx].knn_query(from_state * self.q_mask, k=1)
                    node_idx = node_idx[0][0]
                    node: MultiSearchNode = self.bi_trees[bi_tree_idx][node_idx]

                    best_cost, best_state = self.backward_action_sampler(node, from_state)

                    if best_cost < self.bi_tree_tolerance:
                        state_tuple = (
                            node.state[0] - self.tau_action,
                            best_state,
                            np.zeros_like(node.state[2]),
                            best_state[:self.ctrl_dim].copy()
                        )
                        best_node = MultiSearchNode(
                            node_idx, None, state_tuple, target_config_idx=target_config_idx)
                        
                        self.bi_trees[bi_tree_idx][node_idx]

                        self.bi_trees[bi_tree_idx].append(best_node)
                        knn_item = best_state[1].astype(np.float32) * self.q_mask
                        self.bi_trees_kNNs[bi_tree_idx].add_items(knn_item, ids=[self.trees_kNNs_sizes[bi_tree_idx]])
                        self.bi_trees_kNNs_sizes[bi_tree_idx] += 1
                
                if self.verbose and (i+1) % 1 == 0:
                    fte = np.abs(node.state[1]-best_state).max()
                    print(f"Latest from-to node error: {fte}")
                    print(f"Latest bi-tree node cost: {best_cost}")
                    if self.end_idx != -1:
                        print(f"Bi-tree size at iter {i+1}: {len(self.bi_trees[self.end_idx])}")
                    else:
                        print(f"Bi-trees size at iter {i+1}: {sum([len(tree) for tree in self.bi_trees])}")

            # Store information when appropriate
            if (((self.start_idx == -1) and (i % nodes_per_tree == nodes_per_tree-1)) or
                ((self.start_idx != -1) and (i == self.max_nodes-1))):
                if self.verbose > 3:
                    print(f"Storing tree {start_idx} at i {i}")
                
                self.store_tree(start_idx, folder_path, self.trees)
                
                # Free memory
                self.trees, self.trees_kNNs, self.trees_kNNs_sizes = self.init_trees()

            if self.verbose > 2 and (i+1) % 1000 == 0:
                process = psutil.Process(os.getpid())
                print(f"RSS (resident memory): {process.memory_info().rss / 1024**2:.2f} MB")
                print(f"VMS (virtual memory): {process.memory_info().vms / 1024**2:.2f} MB")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.bidirectional:
            [self.store_tree(i, bi_folder_path, self.bi_trees) for i in range(self.config_count)]
        
        if self.verbose > 1:
            print(f"Total time taken: {total_time:.2f} seconds")

        time_data_path = os.path.join(self.output_dir, f"time_taken.txt")
        with open(time_data_path, "w") as f:
            f.write(f"{total_time}")
