# Define net
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, arch: int=[256, 256]):
        super().__init__()

        self.output_dim = output_dim
        
        layers = [nn.Linear(input_dim, arch[0])]
        
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
        
        layers.append(nn.Linear(arch[-1], output_dim))

        self.linears = nn.ModuleList(layers)

        # init using kaiming
        for layer in self.linears:
            nn.init.kaiming_uniform_(layer.weight)

    def forward(self, current_state: torch.Tensor, goal: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        x = torch.concat([current_state, goal, noise, t], axis=-1)
        for l in self.linears[:-1]:
            x = nn.ReLU()(l(x))
        return self.linears[-1](x)
    

class ActionSamplerDataset(Dataset):
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.goals = []
        self.actions = []

    def add_data(self, state, goal, action):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.goals.pop(0)
            self.actions.pop(0)

        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.goals.append(torch.tensor(goal, dtype=torch.float32))
        self.actions.append(torch.tensor(action, dtype=torch.float32))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.goals[idx], self.actions[idx]


def sample(model: Net, state: torch.Tensor, goal: torch.Tensor, n_samples: int=100, n_steps: int=50, device: str="cpu"):

    model.to(device)
    state = state.unsqueeze(0).expand(n_samples, -1)
    goal = goal.unsqueeze(0).expand(n_samples, -1)
    state = state.to(device)
    goal = goal.to(device)

    x_t = torch.randn((n_samples, model.output_dim)).to(device)
    initial_noise = x_t.clone()

    step_size = 1 / n_steps

    for i in range(n_steps):

        t = torch.full((n_samples, 1), i * step_size, device=device)
        noise_prediction = model(state, goal, x_t, t)
        x_t += step_size * noise_prediction

    return x_t, initial_noise


def train(model: Net, dataset: Dataset, batch_size: int=256, nepochs: int=10, device: str="cpu", verbose: int=0) -> Net:
    
    print(f"Training flow model with {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    prog = tqdm(range(nepochs)) if verbose else range(nepochs)

    for epoch in prog:
        for state, goal, action in dataloader:
            state = state.to(device)
            goal = goal.to(device)
            action = action.to(device)

            optimizer.zero_grad()

            # Forward pass
            t = torch.rand(size=(action.shape[0], 1), device=device)
            noise = torch.randn_like(action)
            noised_action = action * t + noise * (1 - t)

            out = model(state, goal, noised_action, t)
            target = action - noise

            loss = torch.mean((target - out) ** 2)
            losses.append(loss.detach().cpu().item())

            # Backward pass
            loss.backward()
            optimizer.step()

        if verbose and (epoch + 1) % 1000 == 0:
            mean_loss = np.mean(losses)
            losses = []
            print(f"Epoch {epoch+1}, Loss {mean_loss:.6f}")

    return model
