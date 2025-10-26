from tqdm import tqdm
from explore.datasets.utils import build_path


def extract_all_paths(trees: list[list[dict]],
                      min_costs: list[list[float]],
                      top_nodes: list[list[int]],
                      error_thresh: float,
                      horizon: int=4
                      ) -> tuple[list[tuple[int, int]], list[list[dict]], list[list[int]], list[list[int]]]:

    assert horizon >= 1  # How many nodes into the past get optimized for a connection
    tree_count = len(trees)

    # Find graph connections
    config_pairs = []
    paths = []
    connections_mat = [[[] for _ in range(tree_count)] for _ in range(tree_count)]
    connections = [[] for _ in range(tree_count)]
    
    # Initialize connection matrix and connections vector
    for start_idx in range(tree_count):
        for end_idx in range(tree_count):
            if start_idx == end_idx: continue
        
            if min_costs[start_idx][end_idx] < error_thresh:
                connections_mat[start_idx][end_idx].append(end_idx)
                connections[start_idx].append(end_idx)

    # for i in range(tree_count):
    #     print(f"{i}: {connections[i]}\n")

    updated = True
    iteration = 0
    while updated:
        updated = False

        # Loop through every possible starting config and check if a connection to a target config is missing
        for start_idx in range(tree_count):

            connections_updated = connections[start_idx].copy()
            for end_idx in range(tree_count):
                if start_idx == end_idx: continue

                # TODO: Once a path is found the algo will not try to find a shorter path. Fix this.
                if not (end_idx in connections[start_idx]):
                    
                    # Check if the configs which are connected to our starting config have connections to the target config
                    shortest_path = []
                    found_connecter = -1
                    for tree_idx in connections[start_idx]:
                        if (end_idx in connections[tree_idx]):  # Connection found!
                            if (len(shortest_path) == 0 or
                                len(shortest_path) < len(connections_mat[tree_idx][end_idx])):  # Is this path the shortest yet?
                                shortest_path = connections_mat[tree_idx][end_idx].copy()
                                found_connecter = tree_idx
                    
                    if len(shortest_path):
                        connections_mat[start_idx][end_idx] = [found_connecter, *shortest_path]
                        connections_updated.append(end_idx)
                        updated = True
            connections[start_idx] = connections_updated

            iteration += 1
            # print(f"Finished iteration: {iteration}")

    print(f"Finished after {iteration} iterations.")
    
    # for i in range(tree_count):
    #     print(f"{i}: {sorted(connections[i])}\n")

    total_paths = 0
    for i in range(tree_count):
        # print(f"{i}: {connections_mat[i]}\n")
        for c in connections_mat[i]:
            if c: total_paths += 1
    print(f"Real total paths: {total_paths}")

    # Glue paths together
    for start_idx in range(tree_count):
        for end_idx in range(tree_count):
            if start_idx == end_idx: continue

            if len(connections_mat[start_idx][end_idx]):
                path = []
                prev_tree = start_idx
                for tree_idx in connections_mat[start_idx][end_idx]:
                    p = build_path(trees[prev_tree], top_nodes[prev_tree][tree_idx])
                    path.append(p)
                    prev_tree = tree_idx
                
                config_pairs.append((start_idx, end_idx))
                paths.append(path)

    print(f"Avg. Glue Nodes per path: ", sum(len(gn)-1 for gn in paths)/len(paths))
    return config_pairs, paths, connections_mat
