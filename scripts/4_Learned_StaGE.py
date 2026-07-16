import h5py
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from explore.models.TD7 import TD7
from sklearn.neighbors import KDTree
from explore.datasets.utils import build_path
from explore.datasets.StaGE import StaGE, StaGE_Node


def get_tree_successful_nodes(
    tree: list[StaGE_Node],
    all_G_star: np.ndarray,
    min_cost: float
    ) -> tuple[list, list]:
    
    print("Analyzing tree...")
    phis = [node.goal_phi for node in tree]
    sds_tree = KDTree(phis)
    
    end_nodes = []
    reached_targets = []
    for i, manifold_point in tqdm(enumerate(all_G_star), total=len(all_G_star)):
            
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
    q_obj_dot = node.qvel[S.q_obj_dot:S.q_obj_dot+6]
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


def build_path(tree: list[dict], node_idx: int,
               reverse: bool=True) -> list[dict]:

    node = tree[node_idx]
    path = []
    ids = [node_idx]
    
    while True:
        path.append(node)
        if node.parent == -1: break
        node = tree[node.parent]
        ids.append(node.parent)
    
    if reverse:
        path.reverse()
        assert path[0] == tree[0]
    else:
        assert path[0] == tree[node_idx]

    return path, ids


def tree_to_buffer(
    tree: list[StaGE_Node],
    end_nodes: list[int],
    reached_targets: list[int],
    S: StaGE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    states, actions, next_states, rewards, dones = [], [], [], [], []
    success_nodes = []

    def add_path(path, G_target, is_success):
        # compute obs state once per node instead of twice per edge
        obs = [node_obs_state(node, G_target, S) for node in path]
        n_edges = len(path) - 1
        for j in range(n_edges):
            is_last_success_edge = is_success and (j == n_edges - 1)
            states.append(obs[j])
            next_states.append(obs[j + 1])
            actions.append(path[j + 1].ctrl - path[j].ctrl)
            rewards.append(1. if is_last_success_edge else 0.)
            dones.append(1. if is_last_success_edge else 0.)

    print("First pass through...")
    for i, node_id in tqdm(enumerate(end_nodes), total=len(end_nodes)):
        path, ids = build_path(tree, node_id)
        success_nodes.extend(ids)
        add_path(path, S.all_G_star[reached_targets[i]], is_success=True)

    success_size = len(states)
    success_nodes = set(success_nodes)  # O(1) membership checks below

    print("Second pass through...")
    for i in tqdm(range(len(tree) - 1, -1, -1), total=len(tree)):
        if len(states) >= success_size * 2:
            break
        if i in success_nodes:
            continue
        path, _ = build_path(tree, i)
        target = np.random.randint(0, S.manifold_size)
        add_path(path, S.all_G_star[target], is_success=False)

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
    obs = np.broadcast_to(obs, (action_count, obs.shape[0])).copy()
    
    actions = RL_agent.select_action(obs)
    
    return actions


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/Learned_StaGE",
    config_name="humanoid"
)
def main(cfg: DictConfig):

    # Generate initial tree
    file = h5py.File(cfg.configs_path, 'r')
    qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    ctrl = file["ctrl"]

    S = StaGE(qpos, ctrl, cfg.RRT)
    
    # Init agent and environment
    tree = S.init_tree(0)
    obs = node_obs_state(tree[0], S.all_G_star[0], S)
    RL_agent = TD7.Agent(obs.shape[0], S.ctrl_dim, 1., hp=cfg.TD7)
    
    # Main loop
    loop_count = cfg.loop_count
    in_loop_training_steps = cfg.in_loop_training_steps
    agent_sampled_actions = cfg.agent_sampled_actions
    max_loops_before_training = cfg.max_loops_before_training
    
    buffer_full = False
    for i in tqdm(range(loop_count), total=loop_count):
        
        # Generate a new tree with the policy
        print(f"Growing tree {i+1}/{loop_count}...")
        S.start_ids = [np.random.randint(0, S.manifold_size)]
        
        if i < max_loops_before_training and not buffer_full:
            tree = S.run()
        
        else:
            tree = S.run(lambda node, G_star: sample_agent_actions(RL_agent, agent_sampled_actions, node, G_star, S))
        
        # Load tree into buffer
        end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star, S.min_cost)
        print(f"Connection ratio for loop {i+1}/{loop_count}: {(len(reached_targets)/len(S.all_G_star) * 100.):.2f}%")
    
        print(f"Loading tree into buffer {i+1}/{loop_count}...")
        states, actions, next_states, rewards, dones = tree_to_buffer(tree, end_nodes, reached_targets, S)
        
        if len(states):
            RL_agent.replay_buffer.add_multiple(
                states,
                actions,
                next_states,
                rewards.reshape(-1, 1),
                dones.reshape(-1, 1)
            )
            buffer_full = len(RL_agent.replay_buffer) >= RL_agent.replay_buffer.max_size
            print(f"Buffer size: {len(RL_agent.replay_buffer)}")
        
        else:
            print("WARNING: No connections found!")
        
        if i >= max_loops_before_training-1 or buffer_full:
        
            # Train TD7
            print(f"Training {i+1}/{loop_count}...")
            for _ in tqdm(range(in_loop_training_steps), total=in_loop_training_steps):
                RL_agent.train()

    RL_agent.save_checkpoint(path="checkpoints", tag=f"learned_stage")
        

if __name__ == "__main__":
    main()
