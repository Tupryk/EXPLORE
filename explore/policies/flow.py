import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from explore.models.unet import UNet1D
from explore.models.transformer import Transformer


class FlowPolicy(nn.Module):

    def __init__(self,
        obs_dim: int,
        action_dim: int,
        cond_dim: int,
        cfg: DictConfig):
        
        super().__init__()
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.history = cfg.history
        self.horizon = cfg.horizon
        self.denoising_steps = cfg.denoising_steps
        self.effective_horizon = cfg.effective_horizon
        
        if cfg.model_type == "unet":
            self.net = UNet1D(obs_dim, action_dim, cfg.network, cond_dim)
        elif cfg.model_type == "transformer":
            self.net = Transformer(action_dim, self.horizon, self.history+1, obs_dim + cond_dim, cfg.network)
        else:
            raise Exception(f"Moder type '{cfg.model_type}' not implemented yet!")

    def forward(self, obs, goal_cond=None, actions=None):

        if actions is not None:
            batch_size = obs.shape[0]
            
            sample = torch.randn_like(actions)
            timesteps = torch.rand((batch_size, 1, 1))
            noised = sample * (torch.ones_like(timesteps) - timesteps) + actions * timesteps
            
            # pred_noise = self.net(noised, timesteps, obs, goal_cond)
            if goal_cond is not None:
                goal_expanded = goal_cond.unsqueeze(1).expand(-1, obs.size(1), -1)
                obs = torch.cat([obs, goal_expanded], axis=-1)
            
            pred_noise = self.net(noised, timesteps.flatten(), obs)
            pred_actions = sample - pred_noise
            true_path = actions - sample
            
            loss = F.mse_loss(pred_noise, true_path)
            return {"loss": loss, "pred": pred_actions}
        
        else:
            batch_size = obs.shape[0]
            x = torch.randn((batch_size, self.horizon*self.action_dim))
            
            ts = torch.linspace(0.0, 1.0, self.denoising_steps)
            print(ts)
            for t in ts:
                timesteps = torch.full((batch_size,), t)
                if goal_cond is not None:
                    goal_expanded = goal_cond.unsqueeze(1).expand(-1, obs.size(1), -1)
                    obs = torch.cat([obs, goal_expanded], axis=-1)
                
                pred_noise = self.net(x, timesteps, obs)
                x = x - (pred_noise / self.denoising_steps)
            
            denoised_actions = x[:, :self.effective_horizon, :]
            return {"pred": denoised_actions}
