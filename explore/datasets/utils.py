import os
import pickle
import numpy as np


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
