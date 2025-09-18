import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset

from explore.models.vaes import VAE
from experiments.state_encoding.utils import get_state_vectors, eval_state_decoding


states = get_state_vectors("data/15-04-21/trees")
states = np.array(states).astype(np.float32)

print(states)
print(states.shape)

def loss_function(recon_x, x, mu, logvar, epoch, max_beta=.1, warmup_epochs=64):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)  # mean per batch
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    beta = min(max_beta, max_beta * (epoch + 1) / warmup_epochs)
    return BCE + beta * KLD

# Convert to tensor
X_tensor = torch.from_numpy(states)

# Wrap in dataset
dataset = TensorDataset(X_tensor)

# Make shuffled indices
indices = np.random.permutation(len(dataset))

# Compute split sizes
train_size = int(0.8 * len(dataset))
train_idx, test_idx = indices[:train_size], indices[train_size:]

# Create subsets
train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# Make DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)  # smaller batch for evaluation

# Model + optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training
# Training with test set evaluation
for epoch in range(256):
    # --- Training ---
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, epoch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # --- Evaluation ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, epoch)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    
    print(f'Epoch {epoch+1}, Train loss: {avg_train_loss:.4f}, Test loss: {avg_test_loss:.4f}')

torch.save(model.state_dict(), "experiments/state_encoding/model.pth")

model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data[0].to(device)
        recon, mu, logvar = model(data)

        for i in range(min(5, len(data))):
            recon_vec = recon[i].cpu().numpy()
            original_vec = data[i].cpu().numpy()
            eval_state_decoding(original_vec, recon_vec)
        break
