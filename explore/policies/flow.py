import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from explore.models.unet import UNet1D
from explore.datasets.utils import Normalizer
from explore.models.transformer import Transformer


class FlowPolicy(nn.Module):

    def __init__(self,
        obs_dim: int,
        action_dim: int,
        cond_dim: int,
        cfg: DictConfig,
        state_normalizer: Normalizer=None,
        action_normalizer: Normalizer=None,
        device: str="cpu"):
        
        super().__init__()
        
        self.device = device
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.cond_dim = cond_dim
        self.history = cfg.history
        self.horizon = cfg.horizon
        self.denoising_steps = cfg.denoising_steps
        self.effective_horizon = cfg.effective_horizon
        self.state_normalizer = state_normalizer
        self.state_normalizer.to_device(device)
        self.action_normalizer = action_normalizer
        self.action_normalizer.to_device(device)
        
        if cfg.model_type == "transformer":
            self.net = Transformer(action_dim, self.horizon, self.history, obs_dim, cond_dim, cfg.network)
        else:
            raise Exception(f"Moder type '{cfg.model_type}' not implemented yet!")

        self.net.to(device)

    def forward(self, obs, goal_cond=None, actions=None, noise=None):

        if self.action_normalizer is not None and actions is not None:
            actions = self.action_normalizer.normalize(actions)

        if self.state_normalizer != None:
            obs = self.state_normalizer.normalize(obs)
            if goal_cond is not None:
                goal_cond = self.state_normalizer.normalize(goal_cond)

        batch_size = obs.shape[0]
        
        if actions is not None:
            
            sample = torch.randn_like(actions).to(self.device)
            timesteps = torch.rand((batch_size, 1, 1)).to(self.device)
            ones = torch.ones_like(timesteps).to(self.device)
            noised = sample * (ones - timesteps) + actions * timesteps
            
            pred_noise = self.net(noised, timesteps.flatten(), obs, goal_cond)
            true_path = actions - sample
            
            loss = F.mse_loss(pred_noise, true_path)
            return {"loss": loss}
        
        else:
            
            delta = 1.0 / self.denoising_steps
            if noise is not None:
                x = noise.reshape((batch_size, self.horizon, self.action_dim))
            else:
                x = torch.randn((batch_size, self.horizon, self.action_dim)).to(self.device)
            ts = torch.linspace(0.0, 1.0, self.denoising_steps)
            for t in ts:
                timesteps = torch.full((batch_size,), t).to(self.device)
                pred_noise = self.net(x, timesteps, obs, goal_cond)
                x += pred_noise * delta
            
            denoised_actions = x[:, :self.effective_horizon, :]
            if self.action_normalizer is not None:
                denoised_actions = self.action_normalizer.de_normalize(denoised_actions)
            return {"pred": denoised_actions}
