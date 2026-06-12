import os
import time
import copy
import logging
import imageio
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC, TD3
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from explore.models.TD7 import TD7
from explore.env.SGRL_env import StableConfigsEnv


class RL_Trainer:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        
        self.output_dir = cfg.output_dir
        self.total_timesteps = cfg.total_timesteps
        self.timesteps_before_training = cfg.timesteps_before_training
        self.save_as = os.path.join(cfg.output_dir, "final_rl_policy") if cfg.output_dir else "final_rl_policy"
        self.save_freq = cfg.save_freq

        # Set up SB3 logger
        log_dir = os.path.join(cfg.output_dir, "training_logs")
        sb3_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        self.chekpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
        os.makedirs(self.chekpoint_dir)

        # Environment
        self.env = StableConfigsEnv(cfg.env)
        
        # Device
        self.device = cfg.device
        self.logger.info(f"Using device: {self.device}")

        # Model
        self.rl_method = cfg.rl_method

        if self.rl_method == "PPO":
            policy_kwargs = dict(net_arch=cfg.net_arch)
            self.model = PPO(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg.verbose,
                device=self.device
            )

        elif self.rl_method == "SAC":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=cfg.net_arch,
                    qf=cfg.net_arch
                )
            )
            self.model = SAC(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg.verbose,
                device=self.device,
                batch_size=cfg.batch_size
            )

        elif self.rl_method == "TD3":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=cfg.net_arch,
                    qf=cfg.net_arch
                )
            )
            self.model = TD3(
                "MlpPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg.verbose,
                device=self.device,
                batch_size=cfg.batch_size
            )

        elif self.rl_method == "TD7":
            
            eval_cfg = copy.deepcopy(cfg.env)
            eval_cfg.verbose = 0
            eval_cfg.sim_interface.parallel_sims = 1
            self.eval_env = StableConfigsEnv(eval_cfg)
            
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.shape[0] 
            max_action = float(self.env.action_space.high[0])
            
            self.model = TD7.Agent(state_dim, action_dim, max_action)

        else:
            raise Exception(f"RL method '{self.rl_method}' not available.")
        
        if self.rl_method != "TD7":
            self.model.set_logger(sb3_logger)
    
    def train(self):
        self.logger.info("Starting training...")

        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.chekpoint_dir,
            name_prefix=self.rl_method
        )
        
        if self.rl_method != "TD7":
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=checkpoint_callback
            )
        else:
            train_online(self.model, self.env, self.eval_env,
                         max_training_steps=self.total_timesteps,
                         timesteps_before_training=self.timesteps_before_training,
                         output_dir=self.output_dir)

        if self.rl_method != "TD7":
            self.model.save(self.save_as)
        else:
            self.model.agent.save(filename=self.save_as)
        self.logger.info(f"Model saved as {self.save_as}")


def train_online(RL_agent: TD7.Agent, env, eval_env, output_dir: str="", max_training_steps=300000, timesteps_before_training=5000):
    start_time = time.time()
    allow_train = False
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb")) if output_dir else SummaryWriter()
    states, _ = env.reset(done=np.ones(env.sim_count, dtype=bool))
    ep_total_success = np.zeros(env.sim_count)
    ep_total_reward = np.zeros(env.sim_count)
    ep_timesteps = np.zeros(env.sim_count, dtype=int)
    ep_num = 1

    mean_reward_every = 1000
    rewards_count = 0
    success_sum = 0.
    reward_sum = 0.
    success_timesteps_sum = 0
    success_timesteps_count = 0
    fail_timesteps_sum = 0
    fail_timesteps_count = 0

    for t in tqdm(range(max_training_steps), total=max_training_steps):
        
        if output_dir:
            maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, output_dir=output_dir)
        
        if allow_train:
            actions = RL_agent.select_action(np.array(states))
        else:
            actions = np.array([
                env.action_space.sample()
                for _ in range(env.sim_count)
            ])

        next_states, rewards, terminated, truncated, info = env.step(actions)

        ep_total_success += info["goal_reached"]
        ep_total_reward += rewards
        ep_timesteps += 1

        dones_for_buffer = terminated
        dones_for_reset = np.logical_or(terminated, truncated)

        RL_agent.replay_buffer.add_multiple(states, actions, next_states, rewards, dones_for_buffer.astype(float))
        states, _ = env.reset(done=dones_for_reset)
        states[~dones_for_reset] = next_states[~dones_for_reset]

        if allow_train:
                RL_agent.train(writer)

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
                    avg_reward = reward_sum / mean_reward_every

                    print(f"Avg. success rate: {avg_success_rate:.3f}")
                    print(f"Avg. reward: {(reward_sum / mean_reward_every):.3f}")
                    print(f"Avg. success T: {avg_success_t:.1f}")
                    print(f"Avg. fail T: {avg_fail_t:.1f}")
                    print(f"Episodes: {rewards_count}")
                    print(f"Alpha: {env.schedule_alpha:.3f}")

                    writer.add_scalar("rollout/avg_success_rate", avg_success_rate, t)
                    writer.add_scalar("rollout/avg_reward", avg_reward, t)
                    writer.add_scalar("rollout/avg_success_T", avg_success_t, t)
                    writer.add_scalar("rollout/avg_fail_T", avg_fail_t, t)
                    writer.add_scalar("rollout/alpha", env.schedule_alpha, t)
                    
                    success_sum = 0
                    reward_sum = 0

                    success_timesteps_sum = 0
                    success_timesteps_count = 0
                    
                    fail_timesteps_sum = 0
                    fail_timesteps_count = 0

                ep_num += 1
                
            ep_total_success[dones_for_reset] = 0
            ep_total_reward[dones_for_reset] = 0
            ep_timesteps[dones_for_reset] = 0
        
        if (t+1) % 100_000 == 0:
            RL_agent.save_checkpoint(path="checkpoints")

        if t >= timesteps_before_training:
            allow_train = True

    writer.close()


def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, output_dir, eval_freq=25000, eval_eps=20):
    if (t+1) % eval_freq == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
        total_success = np.zeros(eval_eps)
        total_reward = np.zeros(eval_eps)

        for ep in range(eval_eps):
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
                imageio.mimsave(os.path.join(output_dir, f"eval_t{t}_ep{ep}.gif"), frames, fps=24, loop=0)

        print(f"Average total reward over {eval_eps} episodes: {total_reward.mean():.3f} (success rate: {total_success.mean():.3f})")
        print("---------------------------------------")
