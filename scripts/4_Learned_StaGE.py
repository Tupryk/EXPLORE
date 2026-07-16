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
    phis = [node.phi for node in tree]
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


def tree_to_buffer(
    tree: list[StaGE_Node],
    end_nodes: list[int],
    reached_targets: list[int],
    S: StaGE
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    states = []
    actions = []
    next_states = []
    
    rewards = []
    dones = []
    
    success_nodes = []
    for i, node_id in enumerate(end_nodes):
        
        path, ids = build_path(tree, node_id)
        success_nodes.extend(ids)

        prev_node = path[0]
        
        for node in path[1:]:
            
            dones.append(1. if i == len(path)-1 else 0.)
            rewards.append(1. if i == len(path)-1 else 0.)
            
            states.append(
                node_obs_state(prev_node, S.all_G_star[reached_targets[i]], S)
            )
            actions.append(
                node.ctrl - prev_node.ctrl
            )
            next_states.append(
                node_obs_state(node, S.all_G_star[reached_targets[i]], S)
            )
            
            prev_node = node
    
    success_size = len(states)
    for i in range(len(tree)-1, -1, -1):
        if len(states) >= success_size * 2: break
        
        if i in success_nodes:
            continue
        
        path, _ = build_path(tree, node_id)

        prev_node = path[0]
        
        target = np.random.randint(0, S.manifold_size)
        G_target = S.all_G_star[target]
        
        for node in path[1:]:
            
            dones.append(0.)
            rewards.append(0.)
            
            states.append(
                node_obs_state(prev_node, G_target, S)
            )
            actions.append(
                node.ctrl - prev_node.ctrl
            )
            next_states.append(
                node_obs_state(node, G_target, S)
            )
            
            prev_node = node
    
    return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)


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
    config_name="ramp"
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
    loops_before_training = cfg.loops_before_training
    
    for i in tqdm(range(loop_count), total=loop_count):
        
        # Generate a new tree with the policy
        print(f"Growing tree {i+1}/{loop_count}...")
        S.start_ids = [np.random.randint(0, S.manifold_size)]
        
        if i < loops_before_training:
            tree = S.run()
        
        else:
            tree = S.run(lambda node, G_star: sample_agent_actions(RL_agent, agent_sampled_actions, node, G_star))
        
        # Load tree into buffer
        end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star)
        print(f"Connection ratio for loop {i+1}/{loop_count}: {(len(reached_targets)/len(S.all_G_star) * 100.):.2f}%")
    
        print(f"Loading tree into buffer {i+1}/{loop_count}...")
        states, actions, next_states, rewards, dones = tree_to_buffer(tree, end_nodes, reached_targets, S.all_G_star)
        
        RL_agent.replay_buffer.add_multiple(
            states,
            actions,
            next_states,
            rewards.reshape(-1, 1),
            dones.reshape(-1, 1)
        )
        print(f"Buffer size: {len(RL_agent.replay_buffer)}")
        
        if i < loops_before_training:
            continue
        
        # Train TD7
        print(f"Training {i+1}/{loop_count}...")
        for _ in tqdm(range(in_loop_training_steps), total=in_loop_training_steps):
            RL_agent.train()
        

if __name__ == "__main__":
    main()
