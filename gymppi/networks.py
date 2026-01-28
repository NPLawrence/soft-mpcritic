import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class Dynamics(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(self.nx + self.nu, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.nx)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StageCost(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(self.nx + self.nu, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x