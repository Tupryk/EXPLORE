import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=14):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28, 28)
        self.fc11 = nn.Linear(28, 28)
        self.fc12 = nn.Linear(28, 28)
        self.fc21 = nn.Linear(28, latent_dim)   # mean
        self.fc22 = nn.Linear(28, latent_dim)   # log-variance
        self.fc3 = nn.Linear(latent_dim, 28)
        self.fc31 = nn.Linear(28, 28)
        self.fc32 = nn.Linear(28, 28)
        self.fc4 = nn.Linear(28, 13)
        self.fc4_colls = nn.Linear(28, 15)

    def encode(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc11(x))
        h1 = torch.relu(self.fc12(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc31(z))
        h3 = torch.relu(self.fc32(z))
        y, y_colls = self.fc4(h3), torch.sigmoid(self.fc4_colls(h3))
        y = torch.concat([y, y_colls], axis=-1)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
