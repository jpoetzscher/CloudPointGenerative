import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim=3):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        print("ALT MEAN: ", mu.shape)
        print("ALT var: ", log_var.shape)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        print("FAKE DECODER: ", output_dim)
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, z):
        print("FAKE DECODER: ", z.shape)
        h = F.relu(self.fc1(z))
        print("FAKE DECODER: ", z.shape)
        h = F.relu(self.fc2(h))
        print("FAKE DECODER: ", z.shape)
        reconstruction = torch.sigmoid(self.fc3(h))
        print("FAKE DECODER: ", z.shape)
        return reconstruction