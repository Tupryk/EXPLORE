import os
import h5py
import hydra
import pickle
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
    RL_agent = TD7.Agent(obs.shape[0], S.ctrl_dim, 1., hp=cfg.TD7)
    
    # Main loop
    pbar = tqdm(total=int(cfg.min_buffer_size), desc="Filling replay buffer")
    total_trees = 0
    while True:
        # Generate a new tree with the policy
        S.start_ids = [np.random.randint(0, S.manifold_size)]
        tree = S.run()

        # Load tree into buffer
        end_nodes, reached_targets = get_tree_successful_nodes(tree, S.all_G_star, S.min_cost)
        connection_ratio = len(reached_targets) / len(S.all_G_star) * 100.

        # Loading tree into buffer
        states, actions, next_states, rewards, dones = tree_to_buffer(
            tree, end_nodes, reached_targets, S, cfg.failure_ratio
        )

        if len(states) != 0:
            RL_agent.replay_buffer.add_multiple(
                states,
                actions,
                next_states,
                rewards.reshape(-1, 1),
                dones.reshape(-1, 1)
            )
        else:
            tqdm.write(f"WARNING: No connections found in loop {total_trees+1}!")

        pbar.set_postfix({"conn%": f"{connection_ratio:.2f}", "loop": total_trees + 1})
        pbar.n = min(len(RL_agent.replay_buffer), cfg.min_buffer_size)
        pbar.refresh()

        total_trees += 1
        if len(RL_agent.replay_buffer) >= cfg.min_buffer_size:
            break

    stage_buffer_path = os.path.join(cfg.output_dir, f"stage_buffer.pkl")
    with open(stage_buffer_path, "wb") as f:
        pickle.dump(RL_agent.replay_buffer, f)

    pbar.close()
    print("Total trees: ", total_trees)

    print("Warm-starting policy...")
    warm_start_timesteps = int(cfg.min_buffer_size / cfg.TD7.batch_size) * 100
    for _ in tqdm(range(warm_start_timesteps)):
        RL_agent.train()

    env = get_env(cfg)
    states, _ = env.reset(done=np.ones(env.sim_count, dtype=bool))
    ep_total_success = np.zeros(env.sim_count)
    ep_total_reward = np.zeros(env.sim_count)
    ep_timesteps = np.zeros(env.sim_count, dtype=int)

    mean_reward_every = 100
    rewards_count = 0
    success_sum = 0.
    reward_sum = 0.
    success_timesteps_sum = 0
    success_timesteps_count = 0
    fail_timesteps_sum = 0
    fail_timesteps_count = 0

    total_training_steps = int(cfg.total_training_steps)
    for t in tqdm(range(total_training_steps), total=total_training_steps):
        
        actions = RL_agent.select_action(np.array(states))

        next_states, rewards, terminated, truncated, info = env.step(actions)

        ep_total_success += info["goal_reached"]
        ep_total_reward += rewards
        ep_timesteps += 1

        dones_for_buffer = terminated
        dones_for_reset = np.logical_or(terminated, truncated)

        RL_agent.replay_buffer.add_multiple(
            states,
            actions,
            next_states,
            rewards.reshape(-1, 1),
            dones_for_buffer.astype(float).reshape(-1, 1)
        )
        states, _ = env.reset(done=dones_for_reset)
        states[~dones_for_reset] = next_states[~dones_for_reset]

        RL_agent.train()

        if dones_for_reset.any():
            for i in np.where(dones_for_reset)[0]:
                success_sum += ep_total_success[i]
                reward_sum += ep_total_reward[i]
                rewards_count += 1
                
                if ep_total_success[i]:
                    success_timesteps_sum += ep_timesteps[i]
                    success_timesteps_count += 1
                else:
                    fail_timesteps_sum += ep_timesteps[i]
                    fail_timesteps_count += 1
                
                if rewards_count % mean_reward_every == 0:
                    avg_success_t = (success_timesteps_sum / success_timesteps_count) if success_timesteps_count > 0 else float('nan')
                    avg_fail_t = (fail_timesteps_sum / fail_timesteps_count) if fail_timesteps_count > 0 else float('nan')
                    
                    avg_success_rate = success_sum / mean_reward_every

                    print(f"Avg. success rate: {avg_success_rate:.3f}")
                    print(f"Avg. reward: {(reward_sum / mean_reward_every):.3f}")
                    print(f"Avg. success T: {avg_success_t:.1f}")
                    print(f"Avg. fail T: {avg_fail_t:.1f}")
                    print(f"Episodes: {rewards_count}")
                    print(f"Alpha: {env.schedule_alpha:.3f}")

                    success_sum = 0
                    reward_sum = 0

                    success_timesteps_sum = 0
                    success_timesteps_count = 0
                    
                    fail_timesteps_sum = 0
                    fail_timesteps_count = 0

            ep_total_success[dones_for_reset] = 0
            ep_total_reward[dones_for_reset] = 0
            ep_timesteps[dones_for_reset] = 0
        
        if (t+1) % 25000 == 0:
            eval_policy(RL_agent, eval_env, t+1, t, cfg.eval_count, eval_dir)
        
    RL_agent.save_checkpoint(path="checkpoints", tag=f"learned_stage")
    

if __name__ == "__main__":
    main()
