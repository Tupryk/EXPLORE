import os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from explore.datasets.utils import MinMaxNormalizer, load_trees, get_diverse_paths


class ExploreDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 horizon: int=1,
                 history: int=1,
                 min_path_len: int=1,
                 start_idx: int=-1,
                 end_idx: int=-1,
                 verbose: int=0):
        
        # TODO: Vision based
        self.horizon = horizon
        self.history = history
        self.verbose = verbose
        
        config_path = os.path.join(data_dir, ".hydra/config.yaml")
        dataset_cfg = OmegaConf.load(config_path)
        
        tree_dataset = os.path.join(data_dir, "trees")
        self.trees, _, _ = load_trees(tree_dataset)
        
        self.q_mask = np.array(dataset_cfg.RRT.q_mask)
        self.paths, self.traj_pairs = get_diverse_paths(
            self.trees, dataset_cfg.RRT.min_cost,
            self.q_mask, dataset_cfg.RRT.path_diff_thresh,
            min_path_len=min_path_len, cached_folder=data_dir,
            start_idx=start_idx, end_idx=end_idx
        )
        
        if not len(self.traj_pairs):
            raise Exception(f"No feasible trajectories in dataset '{data_dir}'!")
        
        self.q_mask = torch.tensor(dataset_cfg.RRT.q_mask)
    
        self.episode_lengths = []
        self.episode_idxs = []
        self.goal_states = []

        self.states = []
        self.actions = []
        for i, (start_idx, end_idx) in enumerate(self.traj_pairs):
            
            path = self.paths[i]
            path_states = []
            path_actions = []
            for node in path:
                state = node[1].tolist()  # Pos
                # state.extend(node[2].tolist())  # Vel
                path_states.append(state)
                path_actions.append(node[3].tolist())
                
            path_states = torch.tensor(path_states, dtype=torch.float).unsqueeze(1)
            path_actions = torch.tensor(path_actions, dtype=torch.float).unsqueeze(1)

            self.states.append(path_states)
            self.actions.append(path_actions)
            
            traj_len = len(path)
            self.episode_lengths.append(traj_len)
            self.episode_idxs.extend([i for _ in range(traj_len)])
            goal = torch.tensor(self.trees[end_idx][0]["state"][1], dtype=torch.float) * self.q_mask
            self.goal_states.append(goal)
            
        # TODO: Change tau_action
            
        if self.verbose > 0:
            print(f"Total episodes: {len(self.traj_pairs)}.")
            print(f"Avg. length: {sum(self.episode_lengths)/len(self.episode_lengths)} timesteps")
            print(f"Total timesteps: {len(self.episode_idxs)}")
            print(f"Action shape: {self.actions[0][0].shape[1]}")
            print(f"State shape: {self.states[0][0].shape[1]}")
        
        print("Normalizing dataset...")
        min_max_states  = torch.cat(self.states, dim=0)
        min_max_actions = torch.cat(self.actions, dim=0)
        self.action_normalizer = MinMaxNormalizer(min_max_actions[:, 0, :])
        self.state_normalizer = MinMaxNormalizer(min_max_states[:, 0, :])
        
        assert len(self.episode_idxs) == sum(self.episode_lengths)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        episode_idx = self.episode_idxs[idx]
        episode_actions = self.actions[episode_idx]
        episode_states = self.states[episode_idx]
        episode_len = self.episode_lengths[episode_idx]
        timestep = idx - sum(self.episode_lengths[:episode_idx])
        
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

        return action, state, self.goal_states[episode_idx]

    def play_episode(self, idx: int):
        # TODO
        pass
