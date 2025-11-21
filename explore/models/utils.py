import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class Conv1dBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int,
            cond_dim: int,
            kernel_size: int=3,
            n_groups: int=8):
        super().__init__()

        self.conv_block_a = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        self.conv_block_b = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        out = self.conv_block_a(x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        out = out + embed
        out = self.conv_block_b(out)
        out = out + self.residual_conv(x)
        return out
    