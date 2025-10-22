import os
import logging
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

from explore.env.stable_configs_env import StableConfigsEnv


class RL_Trainer:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        
        self.total_timesteps = cfg.total_timesteps
        self.save_as = os.path.join(cfg.output_dir, "trained_rl_policy") if cfg.output_dir else "trained_rl_policy"

        # Set up SB3 logger
        log_dir = os.path.join(cfg.output_dir, "training_logs")
        sb3_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

        # Environment
        self.env = StableConfigsEnv(cfg.env)
        
        # Device
        self.device = cfg.device
        self.logger.info(f"Using device: {self.device}")
        
        # Model

        if cfg.rl_method == "PPO":
            policy_kwargs = dict(net_arch=cfg.net_arch)
            self.model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=cfg.verbose, device=self.device)

        elif cfg.rl_method == "SAC":
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
                device=self.device
            )
            self.model = SAC("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=cfg.verbose, device=self.device)
        
        else:
            raise Exception(f"RL method '{cfg.rl_method}' not available.")
        
        self.model.set_logger(sb3_logger)
    
    def train(self):
        self.logger.info("Starting training...")
        self.model.learn(total_timesteps=self.total_timesteps)
        self.model.save(self.save_as)
        self.logger.info(f"Model saved as {self.save_as}")
