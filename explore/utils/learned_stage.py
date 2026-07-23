import os
import copy
import imageio
import numpy as np
from omegaconf import DictConfig, open_dict

from explore.models.TD7 import TD7
from sklearn.neighbors import KDTree
from explore.env.SGRL_env import StableConfigsEnv
from explore.datasets.StaGE import StaGE, StaGE_Node


def get_tree_successful_nodes(
    tree: list[StaGE_Node],
    all_G_star: np.ndarray,
    min_cost: float
    ) -> tuple[list, list]:
    
    phis = [node.goal_phi for node in tree]
    sds_tree = KDTree(phis)
    
    end_nodes = []
    reached_targets = []
    for i, manifold_point in enumerate(all_G_star):
            
        dist, ind = sds_tree.query([manifold_point], k=1)
        dist = dist[0][0]
        ind = ind[0][0]
            
        if dist < min_cost:
            end_nodes.append(ind)
            reached_targets.append(i)
            
    return end_nodes, reached_targets


def node_obs_state(node: StaGE_Node, G_star: np.ndarray, S: StaGE):
    q = node.qpos[S.q[0]:S.q[1]]
    q_dot = node.qvel[S.q_dot[0]:S.q_dot[1]]
    q_obj_dot = np.concatenate([
        node.qvel[i:i+6] for i in S.q_obj_dot
    ])
    r = node.ctrl
    P = node.geom_xpos[S.P, :].reshape(-1)
    G = node.geom_xpos[S.G, :].reshape(-1)
    
    state = np.concatenate([
        q * S.obs_pos_scale,
        q_dot * S.obs_vel_scale,
        q_obj_dot * S.obs_vel_scale,
        (r - q) * S.obs_ref_err_scale,
        P * S.obs_pos_scale,
        (G_star - G) * S.obs_pos_scale
    ])
    
    return state


def build_path(tree: list[dict], node_idx: int) -> list[dict]:

    node = tree[node_idx]
    path = []
    ids = [node_idx]
    
    while True:
        path.append(node)
        if node.parent == -1: break
        ids.append(node.parent)
        node = tree[node.parent]
    
    path.reverse()
    ids.reverse()
    assert path[0] == tree[0]

    return path, ids


def tree_to_buffer(
    tree: list[StaGE_Node],
    end_nodes: list[int],
    reached_targets: list[int],
    S: StaGE,
    failure_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    states, actions, next_states, rewards, dones = [], [], [], [], []
    success_nodes = []

    def add_path(path, G_target, is_success):
        obs = [node_obs_state(node, G_target, S) for node in path]

        n_edges = len(path) - 1
        for j in range(n_edges):

            is_last_edge = is_success and (j == n_edges - 1)

            states.append(obs[j])
            next_states.append(obs[j + 1])
            actions.append((path[j + 1].ctrl - path[j].ctrl) / S.stepsize)

            rewards.append(1. if is_last_edge else 0.)
            dones.append(1. if is_last_edge else 0.)

    # First pass through (successes + hindsight relabeling)
    for i, node_id in enumerate(end_nodes):
        path, ids = build_path(tree, node_id)
        success_nodes.extend(ids)

        add_path(path, S.all_G_star[reached_targets[i]], is_success=True)

    success_size = len(states)
    success_nodes = set(success_nodes)  # O(1) membership checks below

    # Second pass through (uniform hard-negative sampling)
    non_success_ids = [i for i in range(len(tree)) if i not in success_nodes]

    non_success_ids.extend(end_nodes)  # Avoid bias towards a certain region
    
    n_neg = min(len(non_success_ids), int(success_size * failure_ratio))

    if n_neg > 0:
        chosen = np.random.choice(non_success_ids, size=n_neg, replace=False)

        for i in chosen:
            
            chosen_node = tree[i]
            if chosen_node.parent == -1: continue
            chosen_node_parent = tree[chosen_node.parent]
            chosen_node_G = chosen_node.geom_xpos[S.G, :].reshape(-1)
            chosen_node_parent_G = chosen_node_parent.geom_xpos[S.G, :].reshape(-1)
            
            found = False
            for i in range(50):  # TODO: Make this less ackward maybe?
                random_target_id = np.random.randint(0, S.manifold_size)
                random_target = S.all_G_star[random_target_id]

                dist = np.linalg.norm(random_target - chosen_node_G)
                dist_parent = np.linalg.norm(random_target - chosen_node_parent_G)
                if dist > S.min_cost and dist_parent > S.min_cost:
                    found = True
                    break
            
            if not found: continue
            
            obs = node_obs_state(chosen_node, random_target, S)
            obs_parent = node_obs_state(chosen_node_parent, random_target, S)

            next_states.append(obs)
            states.append(obs_parent)
            actions.append((chosen_node.ctrl - chosen_node_parent.ctrl) / S.stepsize)

            rewards.append(0.)
            dones.append(0.)

    return (np.array(states), np.array(actions), np.array(next_states),
            np.array(rewards), np.array(dones))

def sample_agent_actions(
    RL_agent: TD7.Agent,
    action_count: int,
    node: StaGE_Node,
    G_start: np.ndarray,
    S: StaGE
    ):
    
    obs = node_obs_state(node, G_start, S)
    
    action = RL_agent.select_action(obs.reshape(1, -1), use_exploration=False)
    action_noise = np.random.randn(action_count, S.ctrl_dim) * RL_agent.hp.exploration_noise
    actions = action_noise + action
    
    actions = np.clip(actions, -1, 1)
    return actions


def eval_policy(
    RL_agent: TD7.Agent,
    eval_env: StableConfigsEnv,
    total_timesteps: int,
    loop_id: int,
    eval_count: int,
    output_dir: str
    ):

    print("---------------------------------------")
    print(f"Evaluation at {total_timesteps} time steps")
    total_success = np.zeros(eval_count)
    total_reward = np.zeros(eval_count)

    for ep in range(eval_count):
        state, info = eval_env.reset(options={"alpha": 1.0, "sample_uniform": False, "render": True})
        done = False
        goal_frame = info["goal_frame"]
        frames = []
        while not done:
            action = RL_agent.select_action(np.array(state), use_exploration=False)
            state, rewards, terminated, truncated, info = eval_env.step(action.reshape(1, -1))
            
            frames.extend(info["frames"])
            total_success[ep] += info["goal_reached"][0]
            total_reward[ep] += rewards[0]
            
            done = np.logical_or(terminated[0], truncated[0])

        if frames:
            frames = [(frame.astype(float)*0.8 + goal_frame.astype(float)*0.2).astype(frame.dtype) for frame in frames]
            imageio.mimsave(os.path.join(output_dir, f"eval_t{loop_id+1}_ep{ep+1}.gif"), frames, fps=24, loop=0)

    print(f"Average total reward over {eval_count} episodes: {total_reward.mean():.3f} (success rate: {total_success.mean():.3f})")
    print("---------------------------------------")


def get_eval_env(cfg: DictConfig) -> StableConfigsEnv:

    eval_cfg = copy.deepcopy(cfg.RRT)
    eval_cfg.verbose = 0
    eval_cfg.sim_interface.parallel_sims = 1

    with open_dict(eval_cfg):
        eval_cfg.max_steps = 64
        eval_cfg.stable_configs_path = cfg.configs_path

    eval_env = StableConfigsEnv(eval_cfg)
    return eval_env


def get_env(cfg: DictConfig) -> StableConfigsEnv:

    env_cfg = copy.deepcopy(cfg.RRT)
    env_cfg.verbose = 0
    env_cfg.sim_interface.parallel_sims = 80

    with open_dict(env_cfg):
        env_cfg.use_csrl = True
        env_cfg.schedule_alpha_end_step = 100000
        env_cfg.schedule_alpha_block = 5000
    
        env_cfg.max_steps = 64
        env_cfg.stable_configs_path = cfg.configs_path

    eval_env = StableConfigsEnv(env_cfg)
    return eval_env
