import torch
import torchvision
from torch.distributions import Normal
import numpy as np
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    