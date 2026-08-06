import numpy as np
import torch


class LAP(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        batch_size=256,
        max_action=1,
        normalize_actions=True,
        prioritized=True,
        offline_max_size=None,
        offline_ratio=0.5
    ):
        max_size = int(max_size)
        self.max_size = max_size
        self.offline_max_size = self.max_size if offline_max_size is None else int(offline_max_size)

        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        assert 0. <= offline_ratio <= 1., "offline_ratio must be between 0 and 1"
        self.offline_ratio = offline_ratio

        # ---- Online buffer ----
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        # ---- Offline buffer ----
        self.offline_ptr = 0
        self.offline_size = 0
        self.offline_state = np.zeros((self.offline_max_size, state_dim))
        self.offline_action = np.zeros((self.offline_max_size, action_dim))
        self.offline_next_state = np.zeros((self.offline_max_size, state_dim))
        self.offline_reward = np.zeros((self.offline_max_size, 1))
        self.offline_not_done = np.zeros((self.offline_max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(self.offline_max_size, device=device)
            self.offline_priority = torch.zeros(self.offline_max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

        # index bookkeeping for update_priority
        self.ind = None
        self.offline_ind = None

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action / self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_multiple(self, states, actions, next_states, rewards, dones):
        n = states.shape[0]
        indices = (self.ptr + np.arange(n)) % self.max_size

        self.state[indices] = states
        self.action[indices] = actions / self.normalize_actions
        self.next_state[indices] = next_states
        self.reward[indices] = rewards
        self.not_done[indices] = 1. - dones

        if self.prioritized:
            self.priority[indices] = self.max_priority

        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def add_multiple_to_offline(self, states, actions, next_states, rewards, dones):
        n = states.shape[0]
        indices = (self.offline_ptr + np.arange(n)) % self.offline_max_size

        self.offline_state[indices] = states
        self.offline_action[indices] = actions / self.normalize_actions
        self.offline_next_state[indices] = next_states
        self.offline_reward[indices] = rewards
        self.offline_not_done[indices] = 1. - dones

        if self.prioritized:
            self.offline_priority[indices] = self.max_priority

        self.offline_ptr = (self.offline_ptr + n) % self.offline_max_size
        self.offline_size = min(self.offline_size + n, self.offline_max_size)

    def _sample_indices(self, priority, size, n):
        if self.prioritized:
            csum = torch.cumsum(priority[:size], 0)
            val = torch.rand(size=(n,), device=self.device) * csum[-1]
            ind = torch.searchsorted(csum, val).clamp(0, size - 1).cpu().data.numpy()
        else:
            ind = np.random.randint(0, size, size=n)
        return ind

    def sample(self):
        if self.offline_size > 0:
            n_offline = int(round(self.batch_size * self.offline_ratio))
            n_offline = min(max(n_offline, 0), self.batch_size)
            n_online = self.batch_size - n_offline

            self.ind = self._sample_indices(
                self.priority if self.prioritized else None, self.size, n_online
            )
            self.offline_ind = self._sample_indices(
                self.offline_priority if self.prioritized else None, self.offline_size, n_offline
            )

            state = np.concatenate([self.state[self.ind], self.offline_state[self.offline_ind]], axis=0)
            action = np.concatenate([self.action[self.ind], self.offline_action[self.offline_ind]], axis=0)
            next_state = np.concatenate([self.next_state[self.ind], self.offline_next_state[self.offline_ind]], axis=0)
            reward = np.concatenate([self.reward[self.ind], self.offline_reward[self.offline_ind]], axis=0)
            not_done = np.concatenate([self.not_done[self.ind], self.offline_not_done[self.offline_ind]], axis=0)
        else:
            self.ind = self._sample_indices(
                self.priority if self.prioritized else None, self.size, self.batch_size
            )
            self.offline_ind = None

            state = self.state[self.ind]
            action = self.action[self.ind]
            next_state = self.next_state[self.ind]
            reward = self.reward[self.ind]
            not_done = self.not_done[self.ind]

        return (
            torch.tensor(state, dtype=torch.float, device=self.device),
            torch.tensor(action, dtype=torch.float, device=self.device),
            torch.tensor(next_state, dtype=torch.float, device=self.device),
            torch.tensor(reward, dtype=torch.float, device=self.device),
            torch.tensor(not_done, dtype=torch.float, device=self.device)
        )

    def update_priority(self, priority):
        priority = priority.reshape(-1).detach()

        if self.offline_ind is not None:
            n_online = len(self.ind)
            online_priority = priority[:n_online]
            offline_priority = priority[n_online:]

            candidates = [self.max_priority]
            if len(online_priority) > 0:
                self.priority[self.ind] = online_priority
                candidates.append(float(online_priority.max()))
            if len(offline_priority) > 0:
                self.offline_priority[self.offline_ind] = offline_priority
                candidates.append(float(offline_priority.max()))

            self.max_priority = max(candidates)
        else:
            self.priority[self.ind] = priority
            self.max_priority = max(float(priority.max()), self.max_priority)

    def reset_max_priority(self):
        max_priorities = []
        if self.size > 0:
            max_priorities.append(float(self.priority[:self.size].max()))
        if self.offline_size > 0:
            max_priorities.append(float(self.offline_priority[:self.offline_size].max()))
        self.max_priority = max(max_priorities) if max_priorities else self.max_priority

    def load_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]
        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)

    def __len__(self):
        return self.size + self.offline_size
    