import os
import time
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from explore.models.TD7 import TD7
from explore.env.SGRL_env import StableConfigsEnv


class RL_Trainer:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        
        self.total_timesteps = cfg.total_timesteps
        self.save_as = os.path.join(cfg.output_dir, "final_rl_policy") if cfg.output_dir else "final_rl_policy"
        self.save_freq = cfg.save_freq

        # Set up SB3 logger
        log_dir = os.path.join(cfg.output_dir, "training_logs")
        log_dir = "data/tmp"
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
            # self.eval_env = StableConfigsEnv(cfg.env)
            self.eval_env = None
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
            train_online(self.model, self.env, self.eval_env)

        if self.rl_method != "TD7":
            self.model.save(self.save_as)
        else:
            self.model.agent.save(filename=self.save_as)
        self.logger.info(f"Model saved as {self.save_as}")


def train_online(RL_agent, env, eval_env, max_training_steps=300000, use_checkpoints=True, timesteps_before_training=5000):
    evals = []
    start_time = time.time()
    allow_train = False
    states, _ = env.reset(done=np.ones(env.sim_count, dtype=bool))
    ep_total_reward = np.zeros(env.sim_count)
    ep_timesteps = np.zeros(env.sim_count, dtype=int)
    ep_num = 1

    mean_reward_every = 150
    rewards_count = 0
    reward_sum = 0.

    for t in tqdm(range(max_training_steps), total=max_training_steps):
        # maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time)
        if allow_train:
            actions = RL_agent.select_action(np.array(states))
        else:
            actions = np.array([
                env.action_space.sample()
                for _ in range(env.sim_count)
            ])

        next_states, rewards, dones, _, _ = env.step(actions)
        ep_total_reward += rewards
        ep_timesteps += 1

        RL_agent.replay_buffer.add_multiple(states, actions, next_states, rewards.reshape(-1, 1), dones.astype(float).reshape(-1, 1))
        states, _ = env.reset(done=dones)
        states[~dones] = next_states[~dones]

        if allow_train and not use_checkpoints:
            RL_agent.train()

        if dones.any():
            for i in np.where(dones)[0]:
                # print(f"Total max_training_steps: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps[i]} Reward: {ep_total_reward[i]:.3f} Alpha: {env.schedule_alpha}")
                reward_sum += ep_total_reward[i]
                rewards_count += 1
        
                if (rewards_count+1) % mean_reward_every == 0:
                    print(f"Avg. reward: {(reward_sum / mean_reward_every):.3f}; Episodes: {rewards_count}; Alpha: {env.schedule_alpha}")
                    reward_sum = 0
                
                if allow_train and use_checkpoints:
                    RL_agent.maybe_train_and_checkpoint(ep_timesteps[i], ep_total_reward[i])
                ep_num += 1
        
            ep_total_reward[dones] = 0
            ep_timesteps[dones] = 0
        
        if (t+1) % 100_000 == 0:
            RL_agent.save_checkpoint(path="checkpoints")

        if t >= timesteps_before_training:
            allow_train = True
        

def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, file_name="model", eval_freq=50000, eval_eps=20, use_checkpoints=False):
    if t % eval_freq == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(eval_eps)
        for ep in range(eval_eps):
            state, _ = eval_env.reset(options={"alpha": 1.0})
            done = False
            while not done:
                action = RL_agent.select_action(np.array(state), use_checkpoints, use_exploration=False)
                state, reward, done, _, _ = eval_env.step(action)
                total_reward[ep] += reward

        print(f"Average total reward over {eval_eps} episodes: {total_reward.mean():.3f}")
        print("---------------------------------------")

        evals.append(total_reward)
        np.save(f"./data/{file_name}", evals)
