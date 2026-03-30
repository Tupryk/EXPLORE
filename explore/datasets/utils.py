import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import mujoco
from itertools import combinations
from scipy.spatial.distance import directed_hausdorff, pdist, squareform
from scipy.special import digamma, gamma
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set 

class Normalizer:
    def __init__(self, data: torch.Tensor):
        pass

    def to_device(self, device: str):
        pass

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def de_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

class MinMaxNormalizer(Normalizer):
    def __init__(self, data: torch.Tensor):
        self.mins = data.min(dim=0).values
        self.maxs = data.max(dim=0).values
        self.range = self.maxs - self.mins

        self.range[self.range == 0] = 1.0

    def to_device(self, device: str):
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        self.range = self.range.to(device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * (x - self.mins) / self.range - 1

    def de_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 0.5 * self.range + self.mins

def signum(q1, q2):
    if np.inner(q1, q2)>=0:
        return 1
    else:
        return -1

def cost_computation(node1: dict, node2: dict, q_mask, cost_max_method: bool=False, scene_quat_indices: list=[]) -> float:
    
    state1 = node1["state"][1]
    state2 = node2["state"][1]

    e = (state1 - state2)
    
    for i in scene_quat_indices:
        i0 = i + 4
        e[i:i0] = state1[i:i0] - signum(state1[i:i0], state2[i:i0]) * state2[i:i0]

    e *= q_mask

    if cost_max_method:
        cost = np.abs(e).max()
    else:
        cost = e.T @ e

    return cost

def cost_computation_on_states(state1: np.ndarray, state2: np.ndarray, q_mask, cost_max_method: bool=False, scene_quat_indices: list=[]) -> float:
    
    e = (state1 - state2)
    
    for i in scene_quat_indices:
        i0 = i + 4
        e[i:i0] = state1[i:i0] - signum(state1[i:i0], state2[i:i0]) * state2[i:i0]

    e *= q_mask

    if cost_max_method:
        cost = np.abs(e).max()
    else:
        cost = e.T @ e

    return cost

def load_trees(tree_dataset: str, cutoff: int=-1, verbose: int=0
               ) -> tuple[list[list[dict]], int, int]:
    tree_count = len([n for n in os.listdir(tree_dataset) if ".pkl" in n])
    
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
    check_cached: str="", cost_max_method: bool=False, verbose: int=1) -> tuple[list[list[float]], list[list[int]]]:
    
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
                node_cost = cost_computation(trees[j][0], node, q_mask, cost_max_method)
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
    min_path_len: int=1,
    cached_folder: str="",
    start_idx: int=-1,
    end_idx: int=-1,
    scene_quats: list=[]
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
            path_lens = [len(p) for p in final_paths]
            print("Avg. path length: ", sum(path_lens)/len(path_lens))
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
                node_cost = cost_computation(trees[j][0], node, q_mask, scene_quat_indices=scene_quats)
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
            
            if start_idx != -1 and start_idx != si:
                continue
            
            if end_idx != -1 and end_idx != ei:
                continue

            paths = []
            for end_node in end_nodes[si][ei]:
                fp = build_path(trees[si], end_node, just_states=True)
                if len(fp) >= min_path_len:
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
                            
                            path_diffs[i][j] += cost_computation_on_states(path_a[na][1] - path_b[nb][1], q_mask, scene_quats)
                        
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
                    if len(path) >= min_path_len:
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
    path_lens = [len(p) for p in final_paths]
    print("Avg. path length: ", sum(path_lens)/len(path_lens))
    return final_paths, final_paths_start_end_indices


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


def get_single_tree_reachability(trees, tree_count, q_mask, cost_max_method, ERROR_THRESH, start_idx):
    """
    Computes reachability metadata for ONE specific tree against all target roots.
    """
    # tree_path_count[j] = how many nodes in trees[start_idx] reached root of trees[j]
    tree_path_count = np.zeros(tree_count, dtype=int)
    # tree_end_nodes[j] = list of node indices in trees[start_idx] that reached root of trees[j]
    tree_end_nodes = [[] for _ in range(tree_count)]
    
    # Expensive computation loop (only for the specific start_idx)
    for n_idx, node in enumerate(trees[start_idx]):
        for j in range(tree_count):
            if start_idx == j:
                continue
            
            node_cost = cost_computation(trees[j][0], node, q_mask, cost_max_method)
            if node_cost < ERROR_THRESH:
                tree_path_count[j] += 1
                tree_end_nodes[j].append(n_idx)
                
    return tree_path_count, tree_end_nodes

def compute_coverage_modular(tree_path_count):
    # Coverage: what fraction of other targets were reached at least once?
    coverage = np.count_nonzero(tree_path_count) / (len(tree_path_count) - 1)
    n_paths = np.sum(tree_path_count)
    return coverage, n_paths

def filter_diverse_paths(state_paths, hd_threshold):
    """
    Only keeps paths that are at least 'hd_threshold' away from 
    all other already accepted paths.
    """
    if not state_paths:
        return []

    # approximate maximum ind. set
    # adj_dist_matrix = np.zeros((len(state_paths), len(state_paths)), dtype=bool)
    # for i in range(len(state_paths)):
    #     for j in range(i + 1, len(state_paths)):
    #         is_smaller = hausdorff_distance(state_paths[i], state_paths[j]) < hd_threshold
    #         adj_dist_matrix[i][j] = is_smaller
    #         adj_dist_matrix[j][i] = is_smaller
        
    # G = nx.from_numpy_array(adj_dist_matrix)
    # independent_set = maximum_independent_set(G)
    # diverse_paths = [state_paths[i] for i in independent_set]

    # greedy
    diverse_paths = [state_paths[0]] # Always keep the first found path
    
    for i in range(1, len(state_paths)):
        current_path = state_paths[i]
        # Check against all currently accepted paths
        is_diverse = True
        for accepted in diverse_paths:
            if hausdorff_distance(current_path, accepted) < hd_threshold:
                is_diverse = False
                break
        
        if is_diverse:
            diverse_paths.append(current_path)
            
    return diverse_paths

def compute_metrics_with_diversity(tree, tree_end_nodes, tree_path_count, tree_count, q_mask, hd_threshold=0.05):
    """
    Filters paths by diversity first, then computes Coverage, Hausdorff, and Path Count.
    """
    filtered_path_counts = np.zeros(tree_count, dtype=int)
    explicit_hds = []
    implicit_hds = []

    states_visited = []
    
    for target_idx, count in enumerate(tree_path_count):
        if count == 0:
            continue
            
        raw_state_paths = []
        for node_idx in tree_end_nodes[target_idx]:
            full_path = build_path(tree, node_idx)
            path_states = np.array([n["state"][1]*q_mask for n in full_path])
            raw_state_paths.append(path_states)
            
        diverse_paths = filter_diverse_paths(raw_state_paths, hd_threshold)
        for path in diverse_paths:
            states_visited.extend(path)
        
        num_diverse = len(diverse_paths)
        filtered_path_counts[target_idx] = num_diverse
        
        if num_diverse > 0:
            avg_hd = average_hausdorff_distance(diverse_paths) if num_diverse > 1 else 0.0
            if avg_hd > 0:
                explicit_hds.append(avg_hd)
            implicit_hds.append(avg_hd)

    # Final calculations
    coverage = np.count_nonzero(filtered_path_counts) / (tree_count - 1)
    n_paths = np.sum(filtered_path_counts)
    
    res_hd = np.mean(explicit_hds) if explicit_hds else 0.0
    res_hd_imp = np.mean(implicit_hds) if implicit_hds else 0.0

    ent = compute_entropy(np.array(states_visited)) if states_visited else 0.0
    
    return coverage, n_paths, res_hd, res_hd_imp, ent


def compute_hausdorff_modular(tree, tree_end_nodes, tree_path_count, q_mask):
    """
    Computes Hausdorff metrics based on pre-calculated reachability.
    """
    explicit_scores = []
    implicit_scores = []
    
    for target_idx, count in enumerate(tree_path_count):
        # If path_count is 0, it's unreachable (np.inf in your old code)
        if count == 0:
            continue
            
        # Reconstruct paths for successful connections
        state_paths = []
        for node_idx in tree_end_nodes[target_idx]:
            full_path = build_path(tree, node_idx)
            if len(full_path) <= 2:
                continue
            path_states = np.array([n["state"][1]*q_mask for n in full_path])
            state_paths.append(path_states)
            
        avg_hd = average_hausdorff_distance(state_paths) if len(state_paths) > 1 else 0.0
        
        # Explicit: Only non-zero distances (your original hausdorff_score)
        if avg_hd > 0:
            explicit_scores.append(avg_hd)
        
        # Implicit: Includes 0s for successful paths (your original hausdorff_score_implicit_coverage)
        implicit_scores.append(avg_hd)
    
    # Handle empty cases to avoid division by zero
    res_explicit = np.mean(explicit_scores) if explicit_scores else 0.0
    res_implicit = np.mean(implicit_scores) if implicit_scores else 0.0
    
    return res_explicit, res_implicit

def compute_entropy(states, k=10, n_subsample=100, times=10):
    n, d = states.shape
    print(f"Computing entropy for {n} states in {d}-dimensional space with k={k}...")
    # 1. Guard clause: If we don't have enough points, entropy is undefined/0
    if n <= n_subsample:
        return np.nan
    
    # 2. Adjust k: The maximum possible k-th neighbor is n-1 
    # (since a point cannot be its own neighbor in this context)
    if k >= n:
        return np.nan
    entropy = 0.0
    for _ in range(times):
        states = states[np.random.choice(n, n_subsample, replace=False)]
        n = n_subsample
        dist_matrix = squareform(pdist(states))
        
        # 3. FIX: np.partition(matrix, kth) 
        # If k=2 and n=3, valid indices are 0, 1, 2. 
        # We want the 2nd index (the k-th nearest neighbor).
        # We do NOT use k+1 here; k is the actual index we want to "sort" to that position.
        partitioned_dist = np.partition(dist_matrix, k, axis=1)
        k_nearest_dist = partitioned_dist[:, k]

        # Constants for the estimator
        # Note: Ensure k is at least 1 for gamma(k) and digamma(k)
        k_val = max(1, k)
        c_d = (gamma(d/2 + 1) * np.pi**(d/2)) / gamma(k_val)
        
        entropy += digamma(n) - digamma(k_val) + np.log(c_d) + (d/n) * np.sum(np.log(2 * k_nearest_dist + 1e-10))
    return entropy / times

def compute_path_entropy_modular(tree, tree_end_nodes, tree_path_count, q_mask):
    successful_states = []
    
    for target_idx, count in enumerate(tree_path_count):
        if count == 0:
            continue
            
        for node_idx in tree_end_nodes[target_idx]:
            # Ensure we are handling the state extraction correctly
            state = tree[node_idx]["state"][1] * q_mask
            successful_states.append(state)
            
    # 4. Better guard: compute_entropy needs at least 2 points to find a neighbor
    if len(successful_states) < 2:
        return 0.0
        
    return compute_entropy(np.array(successful_states))
