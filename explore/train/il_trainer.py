import os
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from explore.env.utils import eval_il_policy
from explore.train.utils import warmup_cos_scheduler


class IL_Trainer:
    def __init__(
        self,
        policy: nn.Module,
        dataloader: DataLoader,
        cfg: DictConfig,
        logger: logging.Logger,
        device: str="cpu"
    ):
        self.device = device
        self.checkpoint_every = cfg.checkpoint_every
        self.sim_eval_count = cfg.sim_eval_count
        self.policy = policy.to(self.device)
        self.dataloader = dataloader
        self.optimizer = optim.AdamW(
            policy.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        self.warmup_fraction = cfg.warmup_fraction
        log_dir = os.path.join(cfg.output_dir, "training_logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Make sure checkpoint directory exists
        self.checkpoint_path = os.path.join(cfg.output_dir, "checkpoints")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

    def train(self, epochs, log_interval: int=100, env: gym.Env=None):
        
        warm_up_epochs = int(epochs * self.warmup_fraction)
        scheduler = warmup_cos_scheduler(self.optimizer, warm_up_epochs, epochs)

        global_step = 0
        for epoch in range(1, epochs + 1):
            self.policy.train()
            epoch_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for batch_idx, (actions, obs, goal_cond) in enumerate(pbar):
                actions, obs, goal_cond = actions.to(self.device), obs.to(self.device), goal_cond.to(self.device)
                output = self.policy(obs, goal_cond, actions)
                loss = output["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                if batch_idx % log_interval == 0:
                    self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
                global_step += 1

            avg_loss = epoch_loss / len(self.dataloader)
            self.writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
            tqdm.write(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

            if epoch % self.checkpoint_every == 0 or epoch == epochs:
                
                # Save checkpoint
                name = f"final_policy_epoch_{epoch}" if epoch == epochs else f"epoch_{epoch}"
                ckpt_path = os.path.join(self.checkpoint_path, name)
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(ckpt_path, "model"))
                
                # Eval in sim
                self.policy.eval()
                env_evals_path = os.path.join(ckpt_path, "env_evals")
                os.makedirs(env_evals_path, exist_ok=True)
                if env != None:
                    eval_il_policy(
                        self.policy, env,
                        save_path=env_evals_path,
                        eval_count=self.sim_eval_count,
                        history=self.policy.history
                    )
            
        self.writer.close()
        print(f"Training complete. Model saved to {self.checkpoint_path}")
        