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

def load_trees(tree_dataset: str, cutoff: int=-1, verbose: int=0
               ) -> tuple[list[list[dict]], int, int]:
    tree_count = len(os.listdir(tree_dataset))
    
    if verbose:
        print(f"Tree Count: {tree_count}")
    
    trees = []

    total_nodes_count = 0
    for i in range(tree_count):
        data_path = os.path.join(tree_dataset, f"tree_{i}.pkl")
        with open(data_path, "rb") as f:
            tree: list[dict] = pickle.load(f)
            if cutoff != -1:
                tree = tree[:cutoff]
            trees.append(tree)
            total_nodes_count += len(tree)
    
    return trees, tree_count, total_nodes_count

def build_path(tree: list[dict], node_idx: int,
               just_states: bool=False, reverse: bool=True) -> list[dict]:
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

    if just_states:
        path = [node["state"] for node in path]
    return path

def generate_adj_map( trees: list[list[dict]], q_mask: np.ndarray=np.array([]),
    check_cached: str="", verbose: int=1) -> tuple[list[list[float]], list[list[int]]]:
    
    if check_cached:
        cached_file_path = os.path.join(check_cached, "adj_map.pkl")
        if os.path.exists(cached_file_path):
            if verbose > 0:
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

def get_diverse_paths(
    trees: list[list[dict]],
    error_thresh: float,
    q_mask: np.ndarray,
    diff_thresh: float,
    cached_folder: str="",
    min_len: int=1
    ) -> tuple[list[list[dict]], list[tuple[int, int]]]:
    
    # TODO: There is probably a more efficient way to do this
    tree_count = len(trees)
    
    if cached_folder:
        traj_path = os.path.join(cached_folder, "cached_trajectories")
        if os.path.exists(traj_path):
            final_paths = []
            final_paths_start_end_indices = []
            ns = sorted(os.listdir(traj_path))
            for n in ns:
                cached_file_path = os.path.join(traj_path, n)
                with open(cached_file_path, "rb") as f:
                    a, b = pickle.load(f)
                    final_paths.append(a)
                    final_paths_start_end_indices.append(b)
            
            print(f"Loaded {len(final_paths)} cached trajectories from {traj_path}.")
            return final_paths, final_paths_start_end_indices
    
    # Collect nodes that arrive at a stable configuration
    path_counts = []
    end_nodes = []

    # print("Collecting nodes that arrive at a stable configuration...")
    for i in tqdm(range(tree_count)):

        tree_path_count = [0 for _ in range(tree_count)]
        tree_end_nodes = [[] for _ in range(tree_count)]
        
        for n, node in enumerate(trees[i]):
            for j in range(tree_count):
                node_cost = cost_computation(trees[j][0], node, q_mask)
                if i != j and node_cost < error_thresh:
                    tree_path_count[j] += 1
                    tree_end_nodes[j].append(n)
        
        path_counts.append(tree_path_count)
        end_nodes.append(tree_end_nodes)

    path_counts = np.array(path_counts)
    print(f"Raw total paths found: {path_counts.sum()}, mean: {path_counts.mean()}, std: {path_counts.std()}, max: {path_counts.max()}")
    
    # Calculate difference between paths for the same stable configuration connection
    path_diffs_all = []
    full_paths = [[[] for _ in range(tree_count)] for _ in range(tree_count)]

    print("Calculating difference between paths for the same stable configuration connection...")
    for si in tqdm(range(tree_count)):
        for ei in range(tree_count):
            
            if si == ei or path_counts[si][ei] == 0:
                continue

            paths = []
            for end_node in end_nodes[si][ei]:
                fp = build_path(trees[si], end_node, just_states=True)
                paths.append(fp)
                
            path_count = len(paths)
            path_diffs = [[-1 for _ in range(path_count)] for _ in range(path_count)]
            full_paths[si][ei] = paths

            for i, path_a in enumerate(paths):
                for j, path_b in enumerate(paths):
                    
                    if path_diffs[j][i] == -1:
                        
                        path_diffs[i][j] = 0.0
                        longest_route = max(len(path_a), len(path_b))
                        for n in range(longest_route):
                            na = -1 if n >= len(path_a) else n
                            nb = -1 if n >= len(path_b) else n
                            
                            e = (path_a[na][1] - path_b[nb][1]) * q_mask
                            path_diffs[i][j] += np.sqrt(e.T @ e)
                        
                        path_diffs[i][j] /= longest_route
        
                    else:
                        path_diffs[i][j] = path_diffs[j][i]

            path_diffs = np.array(path_diffs)
            path_diffs_all.append((path_diffs, si, ei))
            
    path_diffs_mat = [[[] for _ in range(tree_count)] for _ in range(tree_count)]
    for pda in path_diffs_all:
        path_diffs_mat[pda[1]][pda[2]] = pda[0]

    # Select diverse paths (paths which are over the difference threshold between each other)
    real_path_counts = []
    trees_paths_idxs = []
    for i in range(tree_count):
        tree_paths = [[] for _ in range(tree_count)]
        tree_path_count = [0 for _ in range(tree_count)]
        
        for j in range(tree_count):
            
            if i != j:
                path_count = len(path_diffs_mat[i][j])
                for p_a in range(path_count):
                    special = True
                    for p_b in tree_paths[j]:
                        if path_diffs_mat[i][j][p_a][p_b] < diff_thresh:
                            special = False
                            break
                        
                    if special:
                        tree_path_count[j] += 1
                        tree_paths[j].append(p_a)
        
        real_path_counts.append(tree_path_count)
        trees_paths_idxs.append(tree_paths)

    real_path_counts = np.array(real_path_counts)
    
    print(f"Real paths found: {real_path_counts.sum()}, mean: {real_path_counts.mean()}, std: {real_path_counts.std()}, max: {real_path_counts.max()}")

    # Format the paths
    final_paths = []
    final_paths_start_end_indices = []
    for si in range(tree_count):
        for ei in range(tree_count):
            if si != ei and len(trees_paths_idxs[si][ei]):
                for path_idx in trees_paths_idxs[si][ei]:
                    path = full_paths[si][ei][path_idx]
                    if len(path) >= min_len:
                        final_paths.append(path)
                        final_paths_start_end_indices.append((si, ei))

    max_config_pairs = tree_count*tree_count-tree_count
    found_pairs_count = 0
    for i in range(tree_count):
        for j in range(tree_count):
            if i != j and real_path_counts[i][j] > 0:
                found_pairs_count += 1
    
    final_paths_count = len(final_paths)
    
    if cached_folder:
        if not os.path.exists(cached_folder):
            os.makedirs(cached_folder)
        traj_path = os.path.join(cached_folder, "cached_trajectories")
        os.makedirs(traj_path)
        for i in range(final_paths_count):
            tn = os.path.join(traj_path, f"trajectory_{i}")
            with open(tn, "wb") as f:
                pickle.dump((final_paths[i], final_paths_start_end_indices[i]), f)
        print(f"Cached {final_paths_count} trajectories in {traj_path}.")
    
    print(f"Total config pairs found: {found_pairs_count} of {max_config_pairs}")
    print(f"Total paths found: {final_paths_count}")
    return final_paths, final_paths_start_end_indices
