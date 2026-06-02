import os
import time
import logging
import numpy as np
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
            self.eval_env = StableConfigsEnv(cfg.env)
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


def train_online(RL_agent, env, eval_env, max_timesteps=300000, use_checkpoints=False, timesteps_before_training=5000):
    evals = []
    start_time = time.time()
    allow_train = False

    state, _ = env.reset()
    ep_finished = False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    for t in range(int(max_timesteps+1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time)
        
        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, ep_finished, _, _ = env.step(action)
        
        ep_total_reward += reward
        ep_timesteps += 1

        done = 1. if ep_finished else 0.
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if allow_train and not use_checkpoints:
            RL_agent.train()

        if ep_finished:
            print(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f} Alpha: {env.schedule_alpha}")

            if allow_train and use_checkpoints:
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= timesteps_before_training:
                allow_train = True

            state, _ = env.reset()
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1
                  

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
