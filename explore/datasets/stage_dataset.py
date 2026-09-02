import torch
import numpy as np
from torch.utils.data import Dataset


class StaGEDataset(Dataset):
    def __init__(self,
                 horizon: int=1,
                 history: int=1,
                 verbose: int=0):
        
        self.verbose = verbose
        self.horizon = horizon
        self.history = history

        self.states = []
        self.actions = []
                
        self.episode_idxs = []
        self.episode_lengths = []
        
        if self.verbose > 0:
            if len(self):
                self.print_stats()
            else:
                print("Empty dataset!")
        
        assert len(self.episode_idxs) == sum(self.episode_lengths)

    def print_stats(self):
        avg_traj_len = sum(self.episode_lengths)/len(self.episode_lengths)
        print(f"Avg. length: {avg_traj_len:.2f} timesteps")
        print(f"Total timesteps: {len(self.episode_idxs)}")
        print(f"Action shape: {self.actions[0][0].shape[1]}")
        print(f"State shape: {self.states[0][0].shape[1]}")

    def add_episode(self, ep_states: list, ep_actions: list):
        ep_states = torch.tensor(np.array(ep_states), dtype=torch.float).unsqueeze(1)
        ep_actions = torch.tensor(np.array(ep_actions), dtype=torch.float).unsqueeze(1)

        self.states.append(ep_states)
        self.actions.append(ep_actions)
        
        traj_len = len(ep_states)
        self.episode_idxs.extend([len(self.episode_lengths) for _ in range(traj_len)])
        self.episode_lengths.append(traj_len)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        episode_idx = self.episode_idxs[idx]
        episode_len = self.episode_lengths[episode_idx]
        timestep = idx - sum(self.episode_lengths[:episode_idx])
        
        episode_actions = self.actions[episode_idx]
        episode_states = self.states[episode_idx]
        
        state = episode_states[timestep]
        
        for i in range(1, self.history):
            idx = timestep-i
            if idx < 0:
                idx = 0
            state = torch.cat((episode_states[idx], state), dim=0)
        
        action = episode_actions[timestep+1] if timestep+1 < episode_len else episode_actions[-1]

        for i in range(1, self.horizon):
            idx = timestep+1+i
            if idx >= episode_len:
                idx = -1
            action = torch.cat((action, episode_actions[idx]), dim=0)

        return action, state

    def state_dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "history": self.history,
            "horizon": self.horizon,
            "episode_idxs": self.episode_idxs,
            "episode_lengths": self.episode_lengths,
        }
