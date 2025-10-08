import os
import cma
import time
import psutil
import pickle
import hnswlib
import numpy as np
from tqdm import trange
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from explore.env.mujoco_sim import MjSim
from explore.utils.utils import randint_excluding


class MultiSearchNode:
    def __init__(self,
                 parent: int,
                 action: np.ndarray,
                 state: tuple,
                 time: float,
                 path: list=None,
                 explore_node: bool=False,
                 target_config_idx: int=-1):
        self.parent = parent
        self.action = action
        self.state = state
        self.time = time
        self.path = path  # Motion
        self.explore_node = explore_node
        self.target_config_idx = target_config_idx

class Search:
    tau_action = 0.1
    tau_sim = 0.01

    def __init__(self, mujoco_xml: str, configs: np.ndarray,
                 configs_ctrl: np.ndarray, cfg: DictConfig):
        
        self.run_name = ""
        self.configs = configs
        self.configs_ctrl = configs_ctrl

        self.max_nodes = int(cfg.max_nodes)
        self.stepsize = cfg.stepsize
        self.target_prob = cfg.target_prob

        self.min_cost = cfg.min_cost
        self.output_dir = cfg.output_dir

        self.sample_count = cfg.sample_count
        self.verbose = cfg.verbose
        self.sample_uniform = cfg.sample_uniform
        self.warm_start = cfg.warm_start
        self.threading = cfg.threading
        self.sampling_strategy = cfg.sampling_strategy
        
        self.start_idx = cfg.start_idx
        self.end_idx = cfg.end_idx
        
        sim_count = 10 if self.threading else 1
        self.sim = [MjSim(mujoco_xml, self.tau_sim, view=False, verbose=0) for _ in range(sim_count)]
        assert (not self.threading) or (self.sample_count % len(self.sim) == 0)
        
        self.ctrl_dim = self.sim[0].data.ctrl.shape[0]
        self.ctrl_ranges = self.sim[0].model.actuator_ctrlrange
        
        if self.sampling_strategy == "rs":
            self.action_sampler = lambda o, t: self.random_sample_ctrls(o, t)
        elif self.sampling_strategy == "cma":
            self.action_sampler = lambda o, t: self.cma_sample_ctrls(o, t)
        else:
            raise Exception(f"Sampling strategy '{self.sampling_strategy}' not implemented yet!")
        
        if self.verbose:
            print(f"Starting search across {self.configs.shape[0]} configs!")

    def reset_trees(self):

        self.trees: list[list[MultiSearchNode]] = []
        self.trees_kNNs: list[hnswlib.Index] = []
        self.trees_kNNs_sizes: list[int] = []
        for i in range(self.configs.shape[0]):
            
            self.sim[0].pushConfig(self.configs[i], self.configs_ctrl[i])
            state = self.sim[0].getState()
        
            root = MultiSearchNode(-1, None, state, 0.)
            self.trees.append([root])

            kNN_tree = hnswlib.Index(space='l2', dim=state[1].shape[0])
            if self.start_idx == -1:
                max_tree_size = int(np.ceil(self.max_nodes / self.configs) + 1)
            else:
                max_tree_size = int(self.max_nodes + self.configs.shape[0])
            kNN_tree.init_index(max_elements=max_tree_size, ef_construction=200, M=16)
            kNN_tree.add_items(state[1].astype(np.float32), ids=[0])

            self.trees_kNNs.append(kNN_tree)
            self.trees_kNNs_sizes.append(1)
    
    def gauss_sample_ctrl(self, parent_node: MultiSearchNode, sample_count: int=1,
                          std_perc: float=0.25) -> np.ndarray:
        # Maybe try sampling over spline waypoints? (technically right now we have a spline with two waypoints)
        mean = parent_node.action if self.warm_start else np.zeros_like(parent_node.action)
        if self.stepsize > 0.:
            std_devs = self.stepsize
        else:
            std_devs = np.abs(self.ctrl_ranges[:, 0] - self.ctrl_ranges[:, 1]) * std_perc
        noise = np.random.randn(sample_count * self.ctrl_dim).reshape(sample_count, -1)
        delta = noise * std_devs + mean
        
        sample = parent_node.state[3] + delta
        return sample
            
    def random_sample_ctrls(self, parent_node: MultiSearchNode, target: tuple
                            ) -> tuple[float, np.ndarray, np.ndarray]:
        
        sampled_ctrls = self.gauss_sample_ctrl(parent_node, self.sample_count)
        
        results = self.eval_multiple_ctrls(sampled_ctrls, parent_node.state, target)
                
        best_node_cost, best_state, best_q = min(results, key=lambda x: x[0])
        return best_node_cost, best_state, best_q
    
    def cma_sample_ctrls(self, parent_node: MultiSearchNode, target: np.ndarray
                         ) -> tuple[float, np.ndarray, np.ndarray]:
        
        pop_size = 20
        initial_guess = np.random.randn(self.ctrl_dim)
        initial_guess = parent_node.state[3]
        es = cma.CMAEvolutionStrategy(initial_guess, 0.5, {
            "popsize": pop_size,
            "maxfevals": self.sample_count,
            "verbose": -1
        })

        while not es.stop():
            candidates = es.ask()

            results = self.eval_multiple_ctrls(candidates, parent_node.state, target)
            fitness_values = [r[0] for r in results]

            es.tell(candidates, fitness_values)
            
            if self.verbose > 3:
                es.disp()
        
        _, best_state, _ = self.eval_ctrl(es.result.xbest, parent_node.state, target)
        return es.result.fbest, best_state, es.result.xbest
    
    def eval_multiple_ctrls(self, ctrls: np.ndarray, origin: tuple,
                            target: tuple) -> np.ndarray:
        results = []
        sim_count = len(self.sim)
        
        if self.threading:
            sample_batch_count = len(ctrls) // sim_count
            for i in range(sample_batch_count):
                # Five Workers and ten simulators seem to be optimal
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(self.eval_ctrl, ctrls[i*sim_count+sim_idx], origin, target, sim_idx)
                        for sim_idx in range(sim_count)
                    ]
                    for future in as_completed(futures):
                        results.append(future.result())
        else:
            for ctrl in ctrls:
                res = self.eval_ctrl(ctrl, origin, target)
                results.append(res)
        
        return results
    
    def eval_ctrl(self, ctrl: np.ndarray, origin: tuple,
                  target: np.ndarray, sim_idx: int=0) -> tuple[float, np.ndarray, np.ndarray]:
        
        self.sim[sim_idx].setState(*origin)
        
        self.sim[sim_idx].step(self.tau_action, ctrl)
        state = self.sim[sim_idx].getState()
        
        e = target - state[1]
        cost2target = e.T @ e
        
        return cost2target, state, ctrl
    
    def store_tree(self, idx: int, folder_path: str):
        dict_tree = []
        for node in self.trees[idx]:
            new_node = {
                "parent": node.parent,
                "action": node.action,
                "state": node.state,
                "time": node.time,
                "path": node.path,
                "explore_node": node.explore_node,
                "target_config_idx": node.target_config_idx
            }
            dict_tree.append(new_node)
            
        data_path = os.path.join(folder_path, f"tree_{idx}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(dict_tree, f)

    def run(self) -> tuple[list[MultiSearchNode], float]:
        
        self.reset_trees()
        
        trees_name = f"{self.run_name}trees"
        folder_path = os.path.join(self.output_dir, trees_name)
        os.makedirs(folder_path, exist_ok=True)
        
        [self.store_tree(i, folder_path) for i, _ in enumerate(self.trees)]

        if self.verbose > 0:
            pbar = trange(self.max_nodes, desc="Physics RRT Search", unit="epoch")
        else:
            pbar = range(self.max_nodes)
            
        nodes_per_tree = self.max_nodes // self.configs.shape[0]
        
        start_time = time.time()
        
        for i in pbar:
            
            if self.start_idx == -1:
                start_idx = i // nodes_per_tree
            else:
                start_idx = self.start_idx

            # Sample random sim state
            exploring = not (np.random.uniform() < self.target_prob) or self.end_idx == -1
            target_config_idx = -1
            
            if exploring or self.end_idx == -1:
                if self.sample_uniform:
                    sim_sample = np.random.uniform(low=-1., high=1., size=self.configs.shape[1])
                else:
                    target_config_idx = randint_excluding(0, self.configs.shape[0], start_idx)  # TODO: exclude end_idx if end_idx != -1
                    sim_sample = self.configs[target_config_idx]
            else:
                target_config_idx = self.end_idx
                sim_sample = self.configs[target_config_idx]
            
            # Pick closest node
            node_idx, _ = self.trees_kNNs[start_idx].knn_query(sim_sample, k=1)
            node_idx = node_idx[0][0]
            node: MultiSearchNode = self.trees[start_idx][node_idx]

            best_node_cost, best_state, best_q = self.action_sampler(node, sim_sample)
                
            best_node = MultiSearchNode(
                node_idx, best_q.copy() - node.action, best_state,  # The copy here is to reduce memory usage: some weird numpy thing about views in arrays or smth...
                node.time + self.tau_action,
                explore_node=exploring,
                target_config_idx=target_config_idx)
            
            self.trees[start_idx].append(best_node)
            self.trees_kNNs[start_idx].add_items(best_state[1].astype(np.float32), ids=[self.trees_kNNs_sizes[start_idx]])
            self.trees_kNNs_sizes[start_idx] += 1
            
            if (((self.start_idx == -1) and (i % nodes_per_tree == nodes_per_tree-1)) or
                ((self.start_idx != -1) and (i == self.max_nodes-1))):
                if self.verbose > 3:
                    print(f"Storing tree {start_idx} at i {i}")
                
                self.store_tree(start_idx, folder_path)
                
                # Free memory
                self.reset_trees()

            if self.verbose > 0 and (i+1) % 1000 == 0:
                process = psutil.Process(os.getpid())
                print(f"RSS (resident memory): {process.memory_info().rss / 1024**2:.2f} MB")
                print(f"VMS (virtual memory): {process.memory_info().vms / 1024**2:.2f} MB")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.verbose > 1:
            print(f"Total time taken: {total_time:.2f} seconds")

        time_data_path = os.path.join(self.output_dir, f"time_taken.txt")
        with open(time_data_path, "w") as f:
            f.write(f"{total_time}")

