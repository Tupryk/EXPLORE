import h5py
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from explore.models.TD7 import TD7
from sklearn.neighbors import KDTree
from explore.datasets.utils import build_path
from explore.env.SGRL_env import StableConfigsEnv
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
    
    for i, node_id in enumerate(end_nodes):
        
        path = build_path(tree, node_id)

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
            
    return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/Learned_StaGE",
    config_name="humanoid_box"
)
def main(cfg: DictConfig):

    # Generate initial tree
    file = h5py.File(cfg.configs_path, 'r')
    qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    ctrl = file["ctrl"]

    S = StaGE(qpos, ctrl, cfg.RRT)
    
    tree = S.run()
    end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star)
    
    print(f"Initial connection ratio: {(len(reached_targets)/len(S.all_G_star) * 100.):.2f}%")
    
    if len(reached_targets) == 0:
        print("Could not find initial connections! Run failed :'(")
        return
    
    # Init agent and environment
    eval_cfg = StableConfigsEnv(cfg.env)
    eval_cfg.verbose = 0
    eval_cfg.sim_interface.parallel_sims = 1
    eval_env = StableConfigsEnv(eval_cfg)
    
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0] 
    max_action = float(eval_env.action_space.high[0])
    
    RL_agent = TD7.Agent(state_dim, action_dim, max_action, hp=cfg.TD7)
    
    # Load tree into buffer
    states, actions, next_states, rewards, dones = tree_to_buffer(tree, end_nodes, reached_targets, S.all_G_star)
    
    RL_agent.replay_buffer.add_multiple(
        states,
        actions,
        next_states,
        rewards.reshape(-1, 1),
        dones.astype(float).reshape(-1, 1)
    )
    
    # Train TD7
    
    # Iterate:
    
    # Get paths from tree and generate a buffer
    
    # Train policy
    
    # Use policy to expand tree
    

if __name__ == "__main__":
    main()
