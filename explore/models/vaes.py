import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(40, 20)
        self.fc21 = nn.Linear(20, latent_dim)   # mean
        self.fc22 = nn.Linear(20, latent_dim)   # log-variance
        self.fc3 = nn.Linear(latent_dim, 20)
        self.fc4 = nn.Linear(20, 40)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
