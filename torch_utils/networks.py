import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def _build_transition_model(env, network_size):
    if network_size == "small":
        return JointMLP_small(env=env)
    if network_size == "medium":
        return JointMLP_medium(env=env)
    if network_size == "large":
        return JointMLP_large(env=env)
    raise ValueError(
        f"Unknown network_size={network_size}. Expected one of 'small', 'medium', 'large'."
    )


class DistributionalDynamicsWrapper(nn.Module):
    """Wrap an existing transition model with a diagonal-Gaussian dynamics head.

    The base model's forward API is preserved for MPPI compatibility, while
    `predict_distribution` provides mean and log-variance for training.
    """

    def __init__(self, base_model, env, hidden_size=256, min_logvar=-10.0, max_logvar=2.0):
        super().__init__()
        self.base_model = base_model
        self.nx = int(np.prod(env.observation_space.shape))
        self.nu = int(np.prod(env.action_space.shape))
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.logvar_net = nn.Sequential(
            nn.Linear(self.nx + self.nu, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.nx),
        )

    def forward(self, x, u):
        return self.base_model(x, u)

    def predict_distribution(self, x, u):
        pred_next_observations, pred_rewards, pred_terminations = self.base_model(x, u)
        raw_logvar = self.logvar_net(torch.cat([x, u], 1))
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * torch.sigmoid(raw_logvar)
        return pred_next_observations, pred_rewards, logvar

    def update_input_stats(self, x, u):
        if hasattr(self.base_model, "update_input_stats"):
            self.base_model.update_input_stats(x, u)

class JointMLP_small(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.get_torch_termination = env.get_wrapper_attr('get_torch_termination')
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
            termination = self.get_torch_termination(x, u, x_next)
        return x_next, reward, termination

class JointMLP_medium(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.get_torch_termination = env.get_wrapper_attr('get_torch_termination')
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
            termination = self.get_torch_termination(x, u, x_next)
        return x_next, reward, termination

class JointMLP_large(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')
        self.get_torch_termination = env.get_wrapper_attr('get_torch_termination')
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
            termination = self.get_torch_termination(x, u, x_next)
        return x_next, reward, termination

class EnsembleDynamicsModel(nn.Module):
    def __init__(self, env, ensemble_size=5, network_size="small"):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}.")

        self.ensemble_size = ensemble_size
        self.network_size = network_size
        self.models = nn.ModuleList(
            [_build_transition_model(env, network_size) for _ in range(ensemble_size)]
        )

    def forward(self, x, u, model_indices=None):
        if model_indices is not None:
            if model_indices.shape[0] != x.shape[0]:
                raise ValueError(
                    f"model_indices must have shape ({x.shape[0]},), got {tuple(model_indices.shape)}."
                )
            model_indices = model_indices.to(device=x.device, dtype=torch.long)
            x_next = torch.empty_like(x)
            reward = None
            termination = None
            for model_idx in torch.unique(model_indices).tolist():
                mask = model_indices == model_idx
                member_x_next, member_reward, member_termination = self.models[model_idx](x[mask], u[mask])
                x_next[mask] = member_x_next
                if reward is None:
                    reward_shape = (x.shape[0],) + tuple(member_reward.shape[1:])
                    reward = torch.empty(reward_shape, device=member_reward.device, dtype=member_reward.dtype)
                if member_termination is not None and termination is None:
                    termination_shape = (x.shape[0],) + tuple(member_termination.shape[1:])
                    termination = torch.empty(termination_shape, device=member_termination.device, dtype=member_termination.dtype)
                reward[mask] = member_reward
                if termination is not None and member_termination is not None:
                    termination[mask] = member_termination
            return x_next, reward, termination

        model_idx = int(torch.randint(self.ensemble_size, (1,), device=x.device).item())
        return self.models[model_idx](x, u)
