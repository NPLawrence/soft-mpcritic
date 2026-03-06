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
        z = torch.cat([x, u], 1)
        y = F.relu(self.fc1(z))
        y = F.relu(self.fc2(y))
        x_next = self.fc3(y)
        return x_next

class StageCost(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(self.nx + self.nu, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y = F.relu(self.fc1(z))
        y = F.relu(self.fc2(y))
        reward = self.fc3(y)
        return reward
    
class JointMLP(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.reward_bounds = self.env.get_wrapper_attr('reward_bounds')
        self.reward_scale = (self.reward_bounds['high'] - self.reward_bounds['low']) / 2
        self.reward_bias = (self.reward_bounds['high'] + self.reward_bounds['low']) / 2

        self.fc1 = nn.Linear(self.nx + self.nu, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.nx + 1)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y = F.relu(self.fc1(z))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        x_next = y[...,:self.nx]
        y_rew = F.tanh(y[...,self.nx:])
        reward = self.reward_scale * y_rew + self.reward_bias
        return x_next, reward
    
class JointMultiMLP(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.reward_bounds = self.env.get_wrapper_attr('reward_bounds')
        self.reward_scale = (self.reward_bounds['high'] - self.reward_bounds['low']) / 2
        self.reward_bias = (self.reward_bounds['high'] + self.reward_bounds['low']) / 2

        self.fc1_dyn = nn.Linear(self.nx + self.nu, 64)
        self.fc2_dyn = nn.Linear(64, 64)
        self.fc3_dyn = nn.Linear(64, self.nx)

        self.fc1_rew = nn.Linear(2*self.nx + self.nu, 64)
        self.fc2_rew = nn.Linear(64, 64)
        self.fc3_rew = nn.Linear(64, 1)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y_dyn = F.relu(self.fc1(z))
        y_dyn = F.relu(self.fc2(y_dyn))
        x_next = self.fc3(y_dyn)

        z_rew = torch.cat([z, x_next], 1)
        y_rew = F.relu(self.fc1(z_rew))
        y_rew = F.relu(self.fc2(y_rew))
        y_rew = F.tanh(self.fc3(y_rew))
        reward = self.reward_scale * y_rew + self.reward_bias

        return x_next, reward