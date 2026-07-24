import os
import h5py
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from explore.models.TD7 import TD7
from explore.datasets.StaGE import StaGE
from explore.utils.learned_stage import *


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/Learned_StaGE",
    config_name="doubleSphere"
)
def main(cfg: DictConfig):

    # Generate initial tree
    file = h5py.File(cfg.configs_path, 'r')
    qpos = file["qpos"] if "qpos" in file.keys() else file["q"]
    ctrl = file["ctrl"]

    eval_dir = os.path.join(cfg.output_dir, "eval_gifs")
    os.makedirs(eval_dir, exist_ok=True)

    eval_env = get_eval_env(cfg)

    S = StaGE(qpos, ctrl, cfg.RRT)
    
    # Init agent and environment
    tree = S.init_tree(0)
    obs = node_obs_state(tree[0], S.all_G_star[0], S)
    RL_agent = TD7.Agent(obs.shape[0], S.ctrl_dim, 1., offline=cfg.TD7.offline_loss, hp=cfg.TD7)
    
    # Main loop
    loop_count = cfg.loop_count
    in_loop_training_steps = cfg.in_loop_training_steps
    agent_sampled_actions = cfg.agent_sampled_actions
    max_loops_before_training = cfg.max_loops_before_training
    pseudo_timesteps = 0
    
    buffer_full = False
    allow_training = False
    for i in tqdm(range(loop_count), total=loop_count):
        
        # Generate a new tree with the policy
        print(f"\nGrowing tree {i+1}/{loop_count} (Using policy: {allow_training})...")
        S.start_ids = [np.random.randint(0, S.manifold_size)]
        
        if not allow_training:
            tree = S.run()
        
        else:
            tree = S.run(lambda node, G_star: sample_agent_actions(RL_agent, agent_sampled_actions, node, G_star, S))
        
        # Load tree into buffer
        end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star, S.min_cost)
        print(f"Connection ratio for loop {i+1}/{loop_count}: {(len(reached_targets)/len(S.all_G_star) * 100.):.2f}%")
    
        # Loading tree into buffer
        states, actions, next_states, rewards, dones = tree_to_buffer(tree, end_nodes, reached_targets, S, cfg.failure_ratio)
        
        if len(states) != 0:
            RL_agent.replay_buffer.add_multiple(
                states,
                actions,
                next_states,
                rewards.reshape(-1, 1),
                dones.reshape(-1, 1)
            )
            buffer_full = len(RL_agent.replay_buffer) >= cfg.min_buffer_size
            print(f"Buffer size: {len(RL_agent.replay_buffer)} (added {len(states)})")
            pseudo_timesteps += len(states)
        
        else:
            print(f"WARNING: No connections found!")
        
        allow_training = i >= max_loops_before_training-1 or buffer_full
        
        # Train TD7
        if allow_training:

            for _ in range(in_loop_training_steps):
                RL_agent.train()
            
            if (i+1) % cfg.eval_freq == 0:
                RL_agent.save_checkpoint(path=eval_dir, tag=f"policy_{i+1}")
                eval_policy(RL_agent, eval_env, pseudo_timesteps, i, cfg.eval_count, eval_dir)

    RL_agent.save_checkpoint(path="checkpoints", tag=f"learned_stage")
        

if __name__ == "__main__":
    main()
