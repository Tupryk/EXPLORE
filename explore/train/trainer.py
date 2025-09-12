import os
import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO

from explore.env.finger_balls_env import FingerBallsEnv


class Trainer:
    def __init__(self, cfg: DictConfig, logger):
        self.cfg = cfg
        self.logger = logger
        
        self.total_timesteps = cfg.total_timesteps

        self.save_as = "trained_rl_policy"
        if cfg.output_dir:
            self.save_as = os.path.join(cfg.output_dir, self.save_as)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        self.env = FingerBallsEnv(cfg.env)

        # Instantiate model, dataset, optimizer, loss
        if cfg.rl_method == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=cfg.verbose)
        else:
            raise Exception(f"RL method '{cfg.rl_method}' not availible.")

    def train(self):
        self.logger.info("Starting training...")
        self.model.learn(total_timesteps=self.total_timesteps)
        self.model.save(self.save_as)
