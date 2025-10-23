import os
import pickle
import numpy as np
from tqdm import tqdm


def cost_computation(node1: dict, node2: dict,
                     q_mask: np.ndarray=np.array([])) -> float:
    
    e = (node1["state"][1] - node2["state"][1])
    if q_mask.shape[0]:
        e *= q_mask
    
    # node_cost = e.T @ e
    cost = np.abs(e).max()
    
    return cost

def load_trees(tree_dataset: str, verbose: int=0) -> tuple[list[list[dict]], int, int]:
    tree_count = len(os.listdir(tree_dataset))
    
    if verbose:
        print(f"Tree Count: {tree_count}")
    
    trees = []

    total_nodes_count = 0
    for i in range(tree_count):
        data_path = os.path.join(tree_dataset, f"tree_{i}.pkl")
        with open(data_path, "rb") as f:
            tree: list[dict] = pickle.load(f)
            trees.append(tree)
            total_nodes_count += len(tree)
    
    return trees, tree_count, total_nodes_count

def build_path(tree: list[dict], node_idx: int, reverse: bool=True) -> list[dict]:
    node = tree[node_idx]
    path = []
    
    while True:
        path.append(node)
        if node["parent"] == -1: break
        node = tree[node["parent"]]
    
    if reverse:
        path.reverse()
        assert path[0] == tree[0]
    else:
        assert path[0] == tree[node_idx]

    return path

def generate_adj_map( trees: list[list[dict]], q_mask: np.ndarray=np.array([]),
    check_cached: str="") -> tuple[list[list[float]], list[list[int]]]:
    
    if check_cached:
        cached_file_path = os.path.join(check_cached, "adj_map.pkl")
        if os.path.exists(cached_file_path):
            print("Loading cached Adjacency Map...")
            with open(cached_file_path, "rb") as f:
                min_costs, top_nodes = pickle.load(f)
            return min_costs, top_nodes

    tree_count = len(trees)
    
    top_nodes = []
    min_costs = []
    
    print("Generating Adjacency Map...")
    for i in tqdm(range(tree_count)):

        tree_min_costs = [float("inf") for _ in range(tree_count)]
        tree_top_nodes = [-1 for _ in range(tree_count)]
        
        for n, node in enumerate(trees[i]):
            for j in range(tree_count):
                node_cost = cost_computation(trees[j][0], node, q_mask)
                if node_cost < tree_min_costs[j]:
                    tree_min_costs[j] = node_cost
                    tree_top_nodes[j] = n
        
        top_nodes.append(tree_top_nodes)
        min_costs.append(tree_min_costs)

    if check_cached:
        with open(cached_file_path, "wb") as f:  # write in binary mode
            pickle.dump((min_costs, top_nodes), f)

    return min_costs, top_nodes

def get_feasible_paths(min_costs: list[list[float]], top_nodes: list[list[int]],
    start_idx: int=-1, end_idx: int=-1, feasible_thresh: float=2e-2,
    ) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    
    tree_count = len(min_costs)
    
    feasible_pairs = []
    end_nodes = []
    best_costs = []
    
    for s in range(tree_count):
        for e in range(tree_count):
            if ((s != e and min_costs[s][e] <= feasible_thresh)
                and (start_idx == -1 or start_idx == s) and (end_idx == -1 or end_idx == e)):
                feasible_pairs.append((s, e))
                end_nodes.append(top_nodes[s][e])
                best_costs.append(min_costs[s][e])
    
    return feasible_pairs, end_nodes, best_costs
