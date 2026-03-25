import os
import datetime
import numpy as np
from tqdm import tqdm
from matplotlib import patches
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from explore.env.mujoco_sim import MjSim
from explore.utils.vis import AdjMap, play_path
from explore.datasets.utils import cost_computation, load_trees, build_path, generate_adj_map, get_feasible_paths


import numpy as np
from scipy.spatial.distance import directed_hausdorff
from itertools import combinations


def comp_hausdorff(trees, tree_count):
    path_counts = []
    end_nodes = []

    for i in tqdm(range(tree_count)):

        tree_path_count = [0 for _ in range(tree_count)]
        tree_end_nodes = [[] for _ in range(tree_count)]
        
        for n, node in enumerate(trees[i]):
            for j in range(tree_count):
                node_cost = cost_computation(trees[j][0], node, q_mask, cost_max_method)
                if i != j and node_cost < ERROR_THRESH:
                    tree_path_count[j] += 1
                    tree_end_nodes[j].append(n)
        
        path_counts.append(tree_path_count)
        end_nodes.append(tree_end_nodes)

    path_counts = np.array(path_counts)

    costs = np.full((tree_count, tree_count), np.inf)

    for si in tqdm(range(tree_count)):
        for ei in range(tree_count):
            
            if path_counts[si][ei] == 0:
                continue

            # Load paths
            paths = []
            for end_node in end_nodes[si][ei]:
                fp = build_path(trees[si], end_node)
                paths.append(fp)
                
            path_count = len(paths)
            
            state_paths = []
            for j in range(path_count):
                tmp_path = []
                for node in paths[j]:
                    tmp_path.append(node["state"][1][:6])
                state_paths.append(np.array(tmp_path))

            avg_hd = average_hausdorff_distance(state_paths)
            print(f"AVG hausdorff for paths between config {si} and {ei}: ",  average_hausdorff_distance(state_paths))
            costs[si, ei] = avg_hd
            print(path_count)
    print(100*"----------")


    hausdorff_score = []
    for i in costs:
        for j in i:
            if j != np.inf and j!=0:
                hausdorff_score.append(j)

    print(np.mean(np.array(hausdorff_score)))

    return np.mean(np.array(hausdorff_score))

    # AdjMap(
    #     costs,
    #     min_value=0.0,
    #     max_value=np.nanmax(costs[np.isfinite(costs)]),
    #     save_as=""
    # )



def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute symmetric Hausdorff distance between two point sets.
    
    A: (N, d)
    B: (M, d)
    """
    d_ab = directed_hausdorff(A, B)[0]
    d_ba = directed_hausdorff(B, A)[0]
    return max(d_ab, d_ba)


def average_hausdorff_distance(state_paths):
    """
    Compute the average pairwise Hausdorff distance
    across a list of paths.
    
    state_paths: list of arrays of shape (Ni, 6)
    """
    if len(state_paths) < 2:
        return 0.0

    distances = []
    
    for A, B in combinations(state_paths, 2):
        d = hausdorff_distance(A, B)
        distances.append(d)

    return float(np.mean(distances))



if __name__ == "__main__":

    dataset = "multirun/2026-02-26/11-38-56/5"
    # dataset = "../multirun/2026-01-31/16-48-06/141"
    # dataset = "../multirun/2026-02-03/16-57-58/0"
    # dataset = "../data/pandasTable_exp"

    config_path = os.path.join(dataset, ".hydra/config.yaml")
    cfg = OmegaConf.load(config_path)

    ERROR_THRESH = cfg.RRT.min_cost
    path_diff_thresh = cfg.RRT.path_diff_thresh
    cost_max_method = False

    look_at_specific_start_idx = cfg.RRT.start_idx
    look_at_specific_end_idx = cfg.RRT.end_idx
    cfg.RRT.start_idx = 0
    look_at_specific_end_idx = -1
    # cutoff = 2500
    cutoff = -1

    q_mask = np.array(cfg.RRT.q_mask)
    sim_cfg = cfg.RRT.sim

    mujoco_xml = os.path.join("..", sim_cfg.mujoco_xml)

    print(f"Looking at start_idx {look_at_specific_start_idx} and end_idx {look_at_specific_end_idx} with error threshold {ERROR_THRESH}.")
    print(f"Tau action: {cfg.RRT.sim.tau_action}; Tau sim: {cfg.RRT.sim.tau_sim}")


    tree_dataset = os.path.join(dataset, "trees")
    trees, tree_count, total_nodes_count = load_trees(tree_dataset, cutoff, verbose=1)

    if not q_mask.shape[0]:
        q_mask = np.ones_like(trees[0][0]["state"][1])

    print("Loaded ", total_nodes_count, " RRT nodes.")

    # print(end_nodes)
    # print(path_counts)
    
    comp_hausdorff(trees, tree_count)
