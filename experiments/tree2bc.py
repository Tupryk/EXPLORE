import numpy as np

from sklearn.neighbors import KDTree
from explore.datasets.StaGE import StaGE, StaGE_Node
from explore.utils.learned_stage import get_tree_successful_nodes, node_obs_state, build_path


def tree_to_buffer(
    tree: list[StaGE_Node],
    end_nodes: list[int],
    reached_targets: list[int],
    S: StaGE,
    failure_ratio: float,
    min_traj_len: float=0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    states, actions = [], [], [], [], []
    success_nodes = []

    def add_path(path, G_target, is_success):
        obs = [node_obs_state(node, G_target, S) for node in path]

        n_edges = len(path) - 1
        for j in range(n_edges):

            is_last_edge = is_success and (j == n_edges - 1)

            states.append(obs[j])
            actions.append((path[j + 1].ctrl - path[j].ctrl) / S.stepsize)

    # Successes
    for i, node_id in enumerate(end_nodes):
        if tree[node_id].t >= min_traj_len:
            path, ids = build_path(tree, node_id)
            success_nodes.extend(ids)

            add_path(path, S.all_G_star[reached_targets[i]], is_success=True)

    return np.array(states), np.array(actions)
    