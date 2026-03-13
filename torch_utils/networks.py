import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class JointMLP_small(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.reward_bounds = env.get_wrapper_attr('reward_bounds')
        self.reward_scale = (self.reward_bounds['high'] - self.reward_bounds['low']) / 2
        self.reward_bias = (self.reward_bounds['high'] + self.reward_bounds['low']) / 2

        self.fc1 = nn.Linear(self.nx + self.nu, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.nx)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y = F.silu(self.fc1(z))
        y = F.silu(self.fc2(y))
        dx = self.fc3(y)
        x_next = x + dx
        with torch.no_grad():
            reward = self.get_torch_reward(x, u, x_next)
        return x_next, reward

class JointMLP_medium(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.reward_bounds = env.get_wrapper_attr('reward_bounds')
        self.reward_scale = (self.reward_bounds['high'] - self.reward_bounds['low']) / 2
        self.reward_bias = (self.reward_bounds['high'] + self.reward_bounds['low']) / 2

        self.fc1 = nn.Linear(self.nx + self.nu, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.nx)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y = F.silu(self.fc1(z))
        y = F.silu(self.fc2(y))
        dx = self.fc3(y)
        x_next = x + dx
        with torch.no_grad():
            reward = self.get_torch_reward(x, u, x_next)
        return x_next, reward

class JointMLP_large(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.reward_bounds = env.get_wrapper_attr('reward_bounds')
        self.reward_scale = (self.reward_bounds['high'] - self.reward_bounds['low']) / 2
        self.reward_bias = (self.reward_bounds['high'] + self.reward_bounds['low']) / 2

        self.fc1 = nn.Linear(self.nx + self.nu, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.nx)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        y = F.silu(self.fc1(z))
        y = F.silu(self.fc2(y))
        dx = self.fc3(y)
        x_next = x + dx
        with torch.no_grad():
            reward = self.get_torch_reward(x, u, x_next)
        return x_next, reward