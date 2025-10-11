import numpy as np


def cost_computation(node1: dict, node2: dict,
                     q_mask: np.ndarray=np.array([])) -> float:
    
    e = (node1["state"][1] - node2["state"][1])
    if q_mask.shape[0]:
        e *= q_mask
    
    # node_cost = e.T @ e
    cost = np.abs(e).max()
    
    return cost