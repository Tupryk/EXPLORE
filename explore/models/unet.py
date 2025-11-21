import torch
import torch.nn as nn
from omegaconf import DictConfig

from explore.models.utils import SinusoidalPosEmb, Conv1dBlock, ConditionalResidualBlock1D

"""
https://github.com/real-stanford/diffusion_policy

This version is modified and broken, don't use it.
"""


class UNet1D(nn.Module):
    def __init__(self, 
        obs_dim: int,
        action_dim: int,
        cfg: DictConfig,
        cond_dim: int=0):
        
        super().__init__()

        self.down_dims = cfg.down_dims
        self.n_groups = cfg.n_groups
        self.kernel_size = cfg.kernel_size
        self.verbose = cfg.verbose
        
        all_dims = [action_dim] + list(self.down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        # Timestep Embedding [(B, 1) -> (B, t_embed)]
        t_embed = cfg.t_embed
        self.timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(t_embed),
            nn.Linear(t_embed, t_embed * 4),
            nn.Mish(),
            nn.Linear(t_embed * 4, t_embed)
        )
        
        # Observation Encoders [(B, obs_dim, history), (B, t_cond_dim) -> (B, ?, ?)]
        t_cond_dim = cond_dim + t_embed
        _, dim_out = in_out[0]
        self.observation_encoder_down = ConditionalResidualBlock1D(
            obs_dim, dim_out, cond_dim=t_cond_dim,
            kernel_size=self.kernel_size, n_groups=self.n_groups
        )
        self.observation_encoder_up = ConditionalResidualBlock1D(
            obs_dim, dim_out, cond_dim=t_cond_dim, 
            kernel_size=self.kernel_size, n_groups=self.n_groups
        )

        # Mid Modules [(B, mid_dim), (B, t_cond_dim) -> (B, ?, ?)]
        mid_module_count = 2
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([])
        for _ in range(mid_module_count):
            module = ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=t_cond_dim,
                kernel_size=self.kernel_size, n_groups=self.n_groups
            )
            self.mid_modules.append(module)

        # Down Modules
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=t_cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=t_cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity()
            ]))

        # Up Modules
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=t_cond_dim,
                    kernel_size=self.kernel_size, n_groups=self.n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=t_cond_dim,
                    kernel_size=self.kernel_size, n_groups=self.n_groups),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))
        
        # Output Layer
        start_dim = self.down_dims[0]
        self.output_layer = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=self.kernel_size),
            nn.Conv1d(start_dim, action_dim, 1),
        )

        if self.verbose > 0:
            print(f"UNet1D parameter count: {sum(p.numel() for p in self.parameters())}")

    def forward(self, 
            sample: torch.Tensor, 
            timesteps: torch.Tensor,
            obs: torch.Tensor,
            cond: torch.Tensor=None):

        sample = sample.permute(0, 2, 1)
        t_cond = self.timestep_encoder(timesteps)

        if cond is not None:
            t_cond = torch.cat([
                t_cond, cond
            ], axis=-1)
        
        # Encode Observations + Conditioning
        obs = obs.permute(0, 2, 1)
        obs_down = self.observation_encoder_down(obs, t_cond)
        obs_up = self.observation_encoder_up(obs, t_cond)
        
        ### Unet Part ###
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, t_cond)
            if idx == 0:
                print("jsajosaiojdsaoijsadoijjio")
                print(x.shape)
                x = x + obs_down
            x = resnet2(x, t_cond)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, t_cond)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_cond)
            print("sakaslk")
            print(x.shape)
            if idx == (len(self.up_modules)-1):
                print("kasoosako")
                print(obs_up.shape)
                x = x + obs_up
            x = resnet2(x, t_cond)
            x = upsample(x)

        x = self.output_layer(x)
        x = x.permute(0, 2, 1)
        return x
    