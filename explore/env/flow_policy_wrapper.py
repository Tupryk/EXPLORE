import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces


class FlowPolicyEnvWrapper(gym.Env):
    def __init__(self, env: gym.Env, policy: nn.Module):
        self.env = env
        self.policy = policy
        self.policy.eval()

        obs_dim = (policy.obs_dim * policy.history) + policy.cond_dim
        action_dim = policy.action_dim * policy.horizon
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(action_dim,), dtype=np.float32)

    def segment_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_in = self.obs[:, :, :-self.policy.cond_dim].clone().float()
        goal_cond = self.obs[:, 0, -self.policy.cond_dim:].clone().float()
        return obs_in, goal_cond
    
    def reset(self, *, seed: int=None, options: dict={}):
        new_obs, info = self.env.reset(seed=seed, options=options)
        
        self.obs = torch.from_numpy(new_obs)
        self.obs = self.obs.expand(1, self.policy.history, self.obs.shape[0])

        obs_in, goal_cond = self.segment_obs()
        obs = torch.cat([obs_in, goal_cond.unsqueeze(0)], dim=1).flatten()

        return obs, info
    
    def step(self, action):
        
        with torch.no_grad():
            obs_in, goal_cond = self.segment_obs()
            noise = torch.from_numpy(action).float()

            actions = self.policy(
                obs_in.to(self.policy.device),
                goal_cond.to(self.policy.device),
                noise=noise.to(self.policy.device)
            )["pred"]
            actions = actions.detach().cpu().numpy()[0]
        
        for a in actions:
            new_obs, reward, terminated, truncated, info = self.env.step(a)
            
            new_obs = torch.from_numpy(new_obs)
            shifted = self.obs.clone()
            shifted[:, :self.policy.history-1, :] = self.obs[:, 1:, :]
            shifted[:, -1, :] = new_obs
            self.obs = shifted.float()
        
        obs_in, goal_cond = self.segment_obs()
        obs = torch.cat([obs_in, goal_cond.unsqueeze(0)], dim=1).flatten()

        return obs, reward, terminated, truncated, info
