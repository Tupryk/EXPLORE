import os
import pickle
import numpy as np
from tqdm import trange
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors

from explore.datasets.adj_map import AdjMap
from explore.env.mujoco_sim import MjSim
from explore.utils.utils import compute_cost_new, randint_excluding


class MultiSearchNode:
    def __init__(self,
                 parent: int,
                 action: np.ndarray,
                 state: tuple,
                 time: float,
                 costs: list,
                 path: list=None,
                 explore_node: bool=False,
                 target_config_idx: int=-1):
        self.parent = parent
        self.action = action
        self.state = state
        self.time = time
        self.path = path  # Motion
        self.explore_node = explore_node
        self.costs = costs
        self.target_config_idx = target_config_idx

class Search:
    tau_action = 0.1
    tau_sim = 0.01

    def __init__(self, configs: np.ndarray, cfg: DictConfig):
        
        self.run_name = ""
        self.configs = configs

        self.sim = MjSim(open("configs/twoFingers.xml", 'r').read(),
                         self.tau_sim, view=False, verbose=0)

        self.max_nodes = cfg.max_nodes
        self.stepsize = cfg.stepsize
        self.target_prob = cfg.target_prob

        self.min_cost = cfg.min_cost
        self.output_dir = cfg.output_dir

        self.sample_count = cfg.sample_count
        self.relevant_frame_names = ["obj", "l_fing", "r_fing"]
        self.relevant_frames_weights = [1., 1., 1.]
        self.relevant_frames_idxs = [0, 1, 2]  # TODO
        self.verbose = cfg.verbose

        self.nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
        
        self.target_states = []

        for i in range(100):
            joint_state = self.configs[i,3:]
            frame_state = np.zeros((1, 7))
            frame_state[0, :3] = self.configs[i,:3]
            self.sim.pushConfig(joint_state, frame_state)
            state = self.sim.getState()
            self.target_states.append(state)

        self.adj_map = AdjMap(self.min_cost, output_dir=self.output_dir)

    def simulate_action(self,
                        q_target: np.ndarray,
                        time_offset: float,
                        display: float=.1):
        
        self.sim.resetSplineRef(time_offset)
        self.sim.setSplineRef(q_target.reshape(1, -1), [self.tau_action], append=False)
        
        self.sim.step(self.tau_action, display)
    
    def getNodeRelevantState(self, tree_idx: int, idx: int) -> np.ndarray:
        node_state = self.trees[tree_idx][idx].state[1][:9]
        return node_state

    def reset_trees(self):

        self.trees: list[list[MultiSearchNode]] = []
        self.trees_kNNs: list[np.ndarray] = []
        for i in range(100):

            joint_state = self.configs[i,3:]
            frame_state = np.zeros((1, 7))
            frame_state[0, :3] = self.configs[i,:3]
            self.sim.pushConfig(joint_state, frame_state)
            state = self.sim.getState()
        
            costs = self.compute_costs(state)
            root = MultiSearchNode(-1, None, state, 0., costs)
            self.trees.append([root])

            kNN_state_list = np.array([self.getNodeRelevantState(i, 0)])
            self.trees_kNNs.append(kNN_state_list)

    def sample_q_target(self, current_q: np.ndarray):
        q_target = current_q + self.stepsize * np.random.randn(current_q.size)
        return q_target

    def compute_costs(self, state: tuple) -> list:
        costs = []
        for target in self.target_states:
            cost = compute_cost_new(state, target)
            costs.append(cost)
        return costs

    def run(self, display: float=1.) -> tuple[list[MultiSearchNode], float]:
        
        self.reset_trees()

        if self.verbose > 0:
            pbar = trange(self.max_nodes, desc="Physics RRT Search", unit="epoch")
        else:
            pbar = range(self.max_nodes)
            
        self.adj_map.save(prefix=f"{self.run_name}start_")

        for _ in pbar:

            # start_idx = np.random.randint(0, 100)
            start_idx = 1
        
            # Fit kNN with current node states
            if len(self.trees[start_idx]) != 1:
                self.trees_kNNs[start_idx] = np.vstack([self.trees_kNNs[start_idx], self.getNodeRelevantState(start_idx, -1)])
            self.nbrs.fit(self.trees_kNNs[start_idx])  # Could possibly be made faster if each new node would not require rebuilding the kNN tree

            # Sample random sim state
            exploring = not (np.random.uniform() < self.target_prob)
            target_config_idx = -1
            if exploring:
                # Sample target sometimes
                sim_sample = np.random.uniform(low=-1., high=1., size=9)
            else:
                target_config_idx = randint_excluding(0, 100, start_idx)
                t = self.target_states[target_config_idx]
                sim_sample = t[1][:9]
            
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
                eval_state = state[1][:9]

                cost2target = np.linalg.norm(sim_sample - eval_state)

                if cost2target < best_node_cost:
                    best_state = self.sim.getState()
                    best_q = q_target
                    best_node_cost = cost2target
            
            costs = self.compute_costs(best_state)
            
            best_node = MultiSearchNode(
                node_idx, best_q, best_state,
                node_start_time + self.tau_action,
                costs, explore_node=exploring,
                target_config_idx=target_config_idx)
            
            self.trees[start_idx].append(best_node)

            # Report
            if self.verbose > 1:
                for i in range(100):
                    if self.trees[start_idx][-1].costs[i] < self.adj_map.costs[start_idx, i]:
                        self.adj_map.set_value(start_idx, i, self.trees[start_idx][-1].costs[i])
                self.adj_map.update_data()
                
                mask = ~np.eye(self.adj_map.costs.shape[0], dtype=bool)
                masked = self.adj_map.costs[mask]
                pbar.set_postfix(avg_cost=masked.mean(), min_cost=masked.min())

        self.adj_map.save(prefix=f"{self.run_name}end_")

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
                    "costs": node.costs,
                    "target_config_idx": node.target_config_idx
                }
                new_tree.append(new_node)
        
            data_path = os.path.join(folder_path, f"tree_{i}.pkl")
            with open(data_path, "wb") as f:
                pickle.dump(new_tree, f)
