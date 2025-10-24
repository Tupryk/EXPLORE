import torch.nn as nn
import torch.nn.functional as F


class FlowNet(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

    def forward(self, obs, actions=None):
        pred_actions = self.net(obs)

        # If expert actions are provided, compute loss
        if actions is not None:
            loss = F.mse_loss(pred_actions, actions)
            return {"loss": loss, "pred": pred_actions}
        else:
            return {"pred": pred_actions}
