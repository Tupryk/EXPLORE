import os
import logging
from omegaconf import DictConfig
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

from explore.models.TD7.TD7 import TD7
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
        policy = "MultiInputPolicy" if cfg.env.use_vision else "MlpPolicy"

        if self.rl_method == "PPO":
            policy_kwargs = dict(net_arch=cfg.net_arch)
            self.model = PPO(
                policy,
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
                policy,
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
                policy,
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=cfg.verbose,
                device=self.device,
                batch_size=cfg.batch_size
            )

        elif self.rl_method == "TD7":
            self.model = TD7(
                self.env,
                cfg.TD7
            )

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
            self.model.learn(
                total_timesteps=self.total_timesteps,
            )

        if self.rl_method != "TD7":
            self.model.save(self.save_as)
        else:
            self.model.agent.save(filename=self.save_as)
        self.logger.info(f"Model saved as {self.save_as}")
