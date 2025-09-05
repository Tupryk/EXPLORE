import os
import torch
import numpy as np
from torch.utils.data import Dataset

from MujocoSim import MjSim
from utils import compute_cost
from rnd_configs import RndConfigs


class FingerBallsDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 horizon: int=1,
                 history: int=0,
                 error_thresh: float=np.nan,
                 state_sigma: float=.0,
                 goal_condition: bool=True,
                 episodes: list[str]=[],
                 flatten_data: bool=False):
        
        self.data_dir = data_dir
        self.files_names = os.listdir(self.data_dir)
        new_file_names = []
        self.horizon = horizon
        self.history = history
        self.error_thresh = error_thresh
        self.flatten_data = flatten_data
        self.state_sigma = state_sigma
        self.goal_condition = goal_condition
        
        relevant_frames = ["obj", "l_fing", "r_fing"]
        De = RndConfigs("data/twoFingers.g", "data/rnd_twoFingers.h5")
        self.relevant_frames_idxs = [De.C.getFrameNames().index(rf) for rf in relevant_frames]
        relevant_frames_weights = [1., 1., 1.]
        
        self.trajectory_lengths = []
        self.episode_idxs = []
        self.goal_states = []

        for name in self.files_names:
            
            if not len(episodes) or name in episodes:

                data_path = os.path.join(self.data_dir, name)
                trajectory_data = np.load(data_path, allow_pickle=True)

                # Check if trajectory is over the error threshold
                target_config_idx = int(name.split('_')[1].split('.')[0])
                De.set_config(target_config_idx)

                error = compute_cost(trajectory_data[-1].state, [De.C.getFrameState()], self.relevant_frames_idxs, relevant_frames_weights)

                if error > self.error_thresh:
                    continue

                # Use path
                traj_len = len(trajectory_data)
                self.trajectory_lengths.append(traj_len)
                self.episode_idxs.extend([len(new_file_names) for _ in range(traj_len)])
                new_file_names.append(name)
                
                if self.goal_condition:
                    sim = MjSim(open('data/twoFingers.xml', 'r').read(), De.C, view=False)
                    goal_state = sim.getState()[0][self.relevant_frames_idxs, :3].flatten()
                    goal_state = torch.tensor(goal_state).float().unsqueeze(0)
                    self.goal_states.append(goal_state)
        
        self.files_names = new_file_names

        print(f"Total episodes: {len(self.files_names)}.\
            Avg. length: {sum(self.trajectory_lengths)/len(self.trajectory_lengths)} timesteps")
        print(f"Total timesteps: {len(self.episode_idxs)}")

    def __len__(self):
        return sum(self.trajectory_lengths)

    def __getitem__(self, idx: int):

        episode_idx = self.episode_idxs[idx]
        name = self.files_names[episode_idx]

        data_path = os.path.join(self.data_dir, name)
        trajectory_data = np.load(data_path, allow_pickle=True)
        
        timestep = idx - sum(self.trajectory_lengths[:episode_idx])
        
        state = trajectory_data[timestep].state[0][self.relevant_frames_idxs, :3].flatten() # Only look at frames that we care about
        state = torch.tensor(state).unsqueeze(0)
            
        for i in range(1, self.history+1):
            idx = timestep-i
            if idx < 0:
                idx = 0
            old_state = trajectory_data[idx].state[0][self.relevant_frames_idxs, :3].flatten()
            old_state = torch.tensor(old_state).unsqueeze(0)
            state = torch.cat((old_state, state), dim=0)
        
        state += torch.randn_like(state) * self.state_sigma

        if self.goal_condition:
            state = torch.cat((self.goal_states[episode_idx], state), dim=0)

        if timestep+1 < len(trajectory_data):
            action = trajectory_data[timestep+1].action
            action = torch.tensor(action).squeeze(0)
            action = action.unsqueeze(0)
        else:
            action = torch.zeros((1, trajectory_data[1].action.shape[-1]))

        for i in range(1, self.horizon):
            idx = timestep+1+i
            if idx < len(trajectory_data):
                new_action = trajectory_data[idx].action
                new_action = torch.tensor(new_action).squeeze(0)
                new_action = new_action.unsqueeze(0)
            else:
                new_action = torch.zeros((1, action.shape[-1]))

            action = torch.cat((action, new_action), dim=0)
        
        if self.flatten_data:
            state = state.flatten().float()
            action = action.flatten().float()
        return state, action
    
    def add(self, states: np.ndarray, actions: np.ndarray):
        
        for episode_idx in range(states.shape[0]):

            ep_states = states[episode_idx]
            ep_actions = actions[episode_idx]

            # TODO: The dataset could potentially be stored as a tree,
            # thereby not replicating a lot of similar starting trajectories
            # that end differently. (This is only relevant when using a history of past states)
            traj_len = ep_states.shape[0]
            self.trajectory_lengths.append(traj_len)
            self.episode_idxs.extend([len(self.trajectory_lengths) for _ in range(traj_len)])

    def play_episode(self, idx: int):
        # TODO
        pass
