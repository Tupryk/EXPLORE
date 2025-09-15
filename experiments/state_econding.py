import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from explore.models.vaes import VAE
from explore.env.mujoco_sim import MjSim


sim = MjSim(open("configs/twoFingers.xml", 'r').read())

trees: list[list[dict]] = []
dataset = "data/15-04-21/trees"

states = []
tree_count = len(os.listdir(dataset))
for i in range(tree_count):
    
    data_path = os.path.join(dataset, f"tree_{i}.pkl")
    with open(data_path, "rb") as f:
        tree: list[dict] = pickle.load(f)

        for node in tree:
            node_state = node["state"]
            sim.setState(*node_state)
            
            pos = node_state[1]
            vel = node_state[2]
            contacts = sim.getContacts()
            
            state = np.hstack((pos, vel, contacts))
            states.append(state)
    
states = np.array(states).astype(np.float32)

print(states)
print(states.shape)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


X_tensor = torch.from_numpy(states)

dataset = TensorDataset(X_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model + optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(128):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}, Average loss: {train_loss/len(train_loader.dataset):.4f}')
