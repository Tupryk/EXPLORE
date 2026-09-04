import os
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from explore.models.TD7 import TD7
from explore.utils.learned_stage import *
from explore.env.single_traj_env import SingleTrajEnv


@hydra.main(
    version_base="1.3",
    config_path="../configs/yaml/Learned_StaGE",
    config_name="humanoidBox"
)
def main(cfg: DictConfig):

    # Load Trajectory
    traj_name = "32_to_48_len_2.60s(53)"
    data_path = "outputs/2026-09-04/10-42-50"

    traj_path = os.path.join(data_path, f"trajs/{traj_name}.pkl")
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    goal_path = os.path.join(data_path, f"goals/{traj_name}.pkl")
    with open(goal_path, "rb") as f:
        goal = pickle.load(f)

    # Environments
    eval_dir = os.path.join(cfg.output_dir, "eval_gifs")
    os.makedirs(eval_dir, exist_ok=True)

    eval_cfg = copy.deepcopy(cfg.RRT)
    eval_cfg.verbose = 0
    eval_cfg.sim_interface.parallel_sims = 1

    with open_dict(eval_cfg):
        eval_cfg.use_curriculum = False
        eval_cfg.max_steps = 64

    eval_env = SingleTrajEnv(goal, traj, eval_cfg, interpolate=True)
    
    env_cfg = copy.deepcopy(cfg.RRT)
    env_cfg.verbose = 0
    env_cfg.sim_interface.parallel_sims = 80

    with open_dict(env_cfg):
        env_cfg.use_curriculum = True
        env_cfg.schedule_alpha_end_step = 50000
        env_cfg.sparse_reward = False
        env_cfg.schedule_alpha_block = 1000
        env_cfg.max_steps = 64

    env = SingleTrajEnv(goal, traj, env_cfg)

    # Agent
    RL_agent = TD7.Agent(env.observation_space.shape[0], env.action_space.shape[0], 1., offline=cfg.TD7.offline_loss, hp=cfg.TD7)

    # Training
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
        
        if (t+1) % cfg.eval_freq == 0:
            eval_policy(RL_agent, eval_env, t+1, t, cfg.eval_count, eval_dir)
        
    RL_agent.save_checkpoint(path="checkpoints", tag=f"learned_stage")
    

if __name__ == "__main__":
    main()
