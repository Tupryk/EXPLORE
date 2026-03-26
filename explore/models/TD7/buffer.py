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
    ):

        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done):
        if state.ndim == 1 or state.shape[0]==1:
            self.state[self.ptr] = state
            self.action[self.ptr] = action / self.normalize_actions
            self.next_state[self.ptr] = next_state
            self.reward[self.ptr] = reward
            self.not_done[self.ptr] = 1.0 - done

            if self.prioritized:
                self.priority[self.ptr] = self.max_priority  # initial priority: maximal

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        else:
            n = state.shape[0]
            self.state[self.ptr : self.ptr + n] = state
            self.action[self.ptr : self.ptr + n] = action / self.normalize_actions
            self.next_state[self.ptr : self.ptr + n] = next_state
            self.reward[self.ptr : self.ptr + n] = reward.reshape(n, 1)
            self.not_done[self.ptr : self.ptr + n] = 1.0 - done.reshape(n, 1)

            if self.prioritized:
                self.priority[self.ptr : self.ptr + n] = self.max_priority  # initial priority: maximal

            self.ptr = (self.ptr + n) % self.max_size
            self.size = min(self.size + n, self.max_size)

    def sample(self, multi=1):
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], 0)
            val = torch.rand(size=(multi * self.batch_size,), device=self.device) * csum[-1]
            indices = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            indices = np.random.randint(0, self.size, size=multi * self.batch_size)

        return (
            indices,
            torch.tensor(self.state[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.action[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[indices], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[indices], dtype=torch.float, device=self.device),
        )

    def update_priority(self, priority, indices):
        prio = priority.reshape(-1).detach()
        self.priority[indices] = prio
        self.max_priority = max(float(prio.max()), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = float(self.priority[: self.size].max())

    def load_D4RL(self, dataset):
        self.state = dataset["observations"]
        self.action = dataset["actions"]
        self.next_state = dataset["next_observations"]
        self.reward = dataset["rewards"].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"].reshape(-1, 1)
        self.size = self.state.shape[0]

        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)


class CloneBuffer(object):
    def __init__(self, state, action, max_size=int(1e6)):
        self.size = state.shape[0]

        self.state = state
        self.action = action

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
        )
