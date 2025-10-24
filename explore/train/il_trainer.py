import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class IL_Trainer:
    def __init__(
        self, model: nn.Module, dataloader: DataLoader,
        cfg: DictConfig, logger: logging.Logger
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()   # Example: behavior cloning on continuous actions
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_path = save_path

        # Make sure checkpoint directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def train(self, epochs: int = 10, log_interval: int = 100):
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for batch_idx, (obs, actions) in enumerate(pbar):
                obs, actions = obs.to(self.device), actions.to(self.device)

                # Forward (includes loss computation)
                output = self.model(obs, actions)
                loss = output["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if batch_idx % log_interval == 0:
                    self.writer.add_scalar("Loss/train_step", loss.item(), global_step)
                global_step += 1

            avg_loss = epoch_loss / len(self.dataloader)
            self.writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
            tqdm.write(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

            torch.save(self.model.state_dict(), self.save_path)

        self.writer.close()
        print(f"Training complete. Model saved to {self.save_path}")
        