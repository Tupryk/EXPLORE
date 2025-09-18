import os
import pickle
import numpy as np
from tqdm import trange
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors

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

    def __init__(self, mujoco_xml: str, configs: np.ndarray, cfg: DictConfig):
        
        self.run_name = ""
        self.configs = configs

        self.sim = MjSim(mujoco_xml, self.tau_sim, view=False, verbose=0)

        self.max_nodes = cfg.max_nodes
        self.stepsize = cfg.stepsize
        self.target_prob = cfg.target_prob

        self.min_cost = cfg.min_cost
        self.output_dir = cfg.output_dir

        self.sample_count = cfg.sample_count
        self.verbose = cfg.verbose
        self.sample_uniform = cfg.sample_uniform
        
        self.start_idx = cfg.start_idx
        self.end_idx = cfg.end_idx

        self.nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
        
    def simulate_action(self,
                        q_target: np.ndarray,
                        time_offset: float,
                        display: float=.1):
        
        self.sim.resetSplineRef(time_offset)
        self.sim.setSplineRef(q_target.reshape(1, -1), [self.tau_action], append=False)
        
        self.sim.step(self.tau_action, display)

    def reset_trees(self):

        self.trees: list[list[MultiSearchNode]] = []
        self.trees_kNNs: list[np.ndarray] = []
        for i in range(self.configs.shape[0]):
            
            self.sim.pushConfig(self.configs[i])
            state = self.sim.getState()
        
            root = MultiSearchNode(-1, None, state, 0.)
            self.trees.append([root])

            kNN_state_list = np.array([state[1]])
            self.trees_kNNs.append(kNN_state_list)

    def sample_q_target(self, current_q: np.ndarray):
        q_target = current_q + self.stepsize * np.random.randn(current_q.size)
        return q_target

    def run(self, display: float=1.) -> tuple[list[MultiSearchNode], float]:
        
        self.reset_trees()

        if self.verbose > 0:
            pbar = trange(self.max_nodes, desc="Physics RRT Search", unit="epoch")
        else:
            pbar = range(self.max_nodes)
            
        for _ in pbar:
            
            start_idx = np.random.randint(0, self.configs.shape[0]) if self.start_idx == -1 else self.start_idx
        
            # Fit kNN with current node states
            self.nbrs.fit(self.trees_kNNs[start_idx])  # Could possibly be made faster if each new node would not require rebuilding the kNN tree

            # Sample random sim state
            exploring = not (np.random.uniform() < self.target_prob) or self.end_idx == -1
            target_config_idx = -1
            if exploring or self.end_idx == -1:
                if self.sample_uniform:
                    # Sample target sometimes
                    sim_sample = np.random.uniform(low=-1., high=1., size=self.configs.shape[1])
                else:
                    target_config_idx = randint_excluding(0, self.configs.shape[0], start_idx)  # TODO: exclude end_idx
                    sim_sample = self.configs[target_config_idx]
            else:
                target_config_idx = self.end_idx
                sim_sample = self.configs[self.end_idx]
            
            # Pick closest node
            _, node_idx = self.nbrs.kneighbors(sim_sample.reshape(1, -1))
            node_idx = int(node_idx[0][0])
            node = self.trees[start_idx][node_idx]

            # Sample random actions in closes node and pick the one closest to the sampled state
            best_node_cost = float("inf")

            node_start_time = node.time
            start_state = node.state
            self.sim.setState(*start_state)
            current_q = start_state[1][:self.sim.data.ctrl.size]

            for _ in range(self.sample_count):
                self.sim.setState(*start_state)

                q_target = self.sample_q_target(current_q)  # This is extremely slow!

                # Simulate for control_tau time
                self.simulate_action(q_target, node_start_time, display)  # This is extremely slow!

                state = self.sim.getState()
                eval_state = state[1]

                cost2target = np.linalg.norm(sim_sample - eval_state)

                if cost2target < best_node_cost:
                    best_state = self.sim.getState()
                    best_q = q_target
                    best_node_cost = cost2target
            
            best_node = MultiSearchNode(
                node_idx, best_q, best_state,
                node_start_time + self.tau_action,
                explore_node=exploring,
                target_config_idx=target_config_idx)
            
            self.trees[start_idx].append(best_node)
            self.trees_kNNs[start_idx] = np.vstack((self.trees_kNNs[start_idx], best_state[1].reshape(1, -1)))

        trees_name = f"{self.run_name}trees"
        folder_path = os.path.join(self.output_dir, trees_name)
        os.makedirs(folder_path, exist_ok=True)

        for i, tree in enumerate(self.trees):
            new_tree = []
            for node in tree:
                new_node = {
                    "parent": node.parent,
                    "action": node.action,
                    "state": node.state,
                    "time": node.time,
                    "path": node.path,
                    "explore_node": node.explore_node,
                    "target_config_idx": node.target_config_idx
                }
                new_tree.append(new_node)
        
            data_path = os.path.join(folder_path, f"tree_{i}.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(new_tree, f)
