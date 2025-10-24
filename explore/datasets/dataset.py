import os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from explore.datasets.utils import load_trees, generate_adj_map, get_feasible_paths, build_path


class ExploreDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 horizon: int=1,
                 history: int=0,
                 state_sigma: float=.0,
                 goal_condition: bool=True,
                 flatten_data: bool=True,
                 verbose: int=0):
        
        # TODO: Vision based
        self.horizon = horizon
        self.history = history
        self.flatten_data = flatten_data
        self.state_sigma = state_sigma
        self.goal_condition = goal_condition
        self.verbose = verbose
        
        config_path = os.path.join(data_dir, ".hydra/config.yaml")
        dataset_cfg = OmegaConf.load(config_path)
        
        tree_dataset = os.path.join(data_dir, "trees")
        self.trees, _, _ = load_trees(tree_dataset)
        
        self.q_mask = np.array(dataset_cfg.RRT.q_mask)
        min_costs, top_nodes = generate_adj_map(self.trees, self.q_mask, check_cached=data_dir)
        self.traj_pairs, self.traj_end_nodes, _ = get_feasible_paths(min_costs, top_nodes,
            dataset_cfg.RRT.start_idx, dataset_cfg.RRT.end_idx, dataset_cfg.RRT.min_cost)
        
        if not len(self.traj_pairs):
                raise Exception(f"No feasible trajectories in dataset '{data_dir}'!")
        
        self.q_mask = torch.tensor(dataset_cfg.RRT.q_mask)
    
        self.episode_lengths = []
        self.episode_idxs = []
        self.goal_states = []

        self.paths = []
        for i, (start_idx, end_idx) in enumerate(self.traj_pairs):
            
            path = build_path(self.trees[start_idx], self.traj_end_nodes[i])
            self.paths.append(path)
            
            traj_len = len(path)
            self.episode_lengths.append(traj_len)
            self.episode_idxs.extend([i for _ in range(traj_len)])
            self.goal_states.append(torch.tensor(self.trees[end_idx][0]["state"][1]))
            
        if self.verbose > 0:
            print(f"Total episodes: {len(self.traj_pairs)}.")
            print(f"Avg. length: {sum(self.episode_lengths)/len(self.episode_lengths)} timesteps")
            print(f"Total timesteps: {len(self.episode_idxs)}")
        
        assert len(self.episode_idxs) == sum(self.episode_lengths)

    def __len__(self):
        return sum(self.episode_lengths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        episode_idx = self.episode_idxs[idx]
        episode_data = self.paths[episode_idx]
        timestep = idx - sum(self.episode_lengths[:episode_idx])
        
        # TODO: use_vel
        state = torch.tensor(episode_data[timestep]["state"][1])
            
        for i in range(1, self.history+1):
            idx = timestep-i
            if idx < 0:
                idx = 0
            old_state = torch.tensor(episode_data[idx]["state"][1])
            state = torch.cat((old_state, state), dim=0)
        
        state += torch.randn_like(state) * self.state_sigma

        if self.goal_condition:
            state = torch.cat((self.goal_states[episode_idx], state), dim=0)

        if timestep+1 < len(episode_data):
            # TODO: delta_q?
            action = episode_data[timestep+1]["delta_q"]
            action = torch.tensor(action).squeeze(0)
            action = action.unsqueeze(0)
        else:
            action = torch.zeros((1, episode_data[1]["delta_q"].shape[-1]))

        for i in range(1, self.horizon):
            idx = timestep+1+i
            if idx < len(episode_data):
                new_action = episode_data[idx]["delta_q"]
                new_action = torch.tensor(new_action).squeeze(0)
                new_action = new_action.unsqueeze(0)
            else:
                new_action = torch.zeros((1, action.shape[-1]))

            action = torch.cat((action, new_action), dim=0)
        
        if self.flatten_data:
            state = state.flatten().float()
            action = action.flatten().float()
        return state, action

    def play_episode(self, idx: int):
        # TODO
        pass
