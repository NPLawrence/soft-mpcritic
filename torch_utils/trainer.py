import torch
import torch.nn as nn
from torch.optim.adam import Adam
from typing import Any


def gaussian_nll_diag(pred_mean, target, pred_logvar):
    inv_var = torch.exp(-pred_logvar)
    return 0.5 * torch.mean(inv_var * torch.square(target - pred_mean) + pred_logvar)

class NullScaler(nn.Module):
    def __init__(self, nx: int):
        super().__init__()

    def fit(self, observations: torch.Tensor):
        self.fitted = torch.tensor(True)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled

class MinMaxScaler(nn.Module):
    """Min-max scaler that maps observations to [-1, 1].
    
    Call `fit(observations)` once before training, then the scaler
    is applied automatically in the member model's forward pass.
    """
    def __init__(self, nx: int, epsilon: float = 1e-6):
        super().__init__()
        # Register as buffers so they move with .to(device) and are saved in state_dict
        self.epsilon = epsilon
        self.register_buffer("obs_min", torch.zeros(nx))
        self.register_buffer("obs_max", torch.ones(nx))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, observations: torch.Tensor):
        """Compute per-feature min/max from a (N, nx) tensor."""
        self.obs_min = observations.min(axis=0).values
        self.obs_max = observations.max(axis=0).values
        self.fitted = torch.tensor(True)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        range_ = (self.obs_max - self.obs_min).clamp(min=self.epsilon)
        return 2.0 * (x - self.obs_min) / range_ - 1.0          # → [-1, 1]

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        range_ = (self.obs_max - self.obs_min).clamp(min=self.epsilon)
        return (x_scaled + 1.0) / 2.0 * range_ + self.obs_min   # → original space

class StandardScaler(nn.Module):
    def __init__(self, nx: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(nx))
        self.register_buffer("std",  torch.ones(nx))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, observations: torch.Tensor):
        self.mean   = observations.mean(axis=0)
        self.std    = observations.std(axis=0).clamp(min=self.epsilon)
        self.fitted = torch.tensor(True)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self.std + self.mean

scaler_map = {
    'null' : NullScaler,
    'minmax' : MinMaxScaler,
    'standard': StandardScaler,
}

class Trainer():
    def __init__(
        self,
        model,
        optimizer_class: Any = Adam,
        lr=3e-4,
        model_loss=None,
        huber_delta=1.0,
        scaler='null',
    ):
        self.model = model
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)
        self.scaler = scaler_map[scaler](model.nx)

    def update(self, data):
        device = next(self.model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        if hasattr(self.model, "update_input_stats"):
            self.model.update_input_stats(observations, actions)

        if hasattr(self.model, "predict_distribution"):
            pred_next_observations, pred_rewards, pred_logvar = self.model.predict_distribution(observations, actions)
            pred_dynamics = pred_next_observations - observations
            target_dynamics = next_observations - observations
            dynamics_loss = gaussian_nll_diag(pred_dynamics, target_dynamics, pred_logvar)
        else:
            pred_next_observations, pred_rewards, pred_terminations = self.model(observations, actions)
            pred_dynamics = pred_next_observations - observations
            target_dynamics = next_observations - observations
            scaled_pred_dynamics = self.scaler.scale(pred_dynamics)
            scaled_target_dynamics = self.scaler.scale(target_dynamics)
            dynamics_loss = self.model_loss(scaled_pred_dynamics, scaled_target_dynamics)
        reward_loss = self.model_loss(pred_rewards, rewards)
        loss = dynamics_loss + reward_loss
        
        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return dynamics_loss, reward_loss
    

class Trainer_ValueAligned():
    def __init__(
        self,
        model,
        actor,
        critic,
        gamma=0.99,
        T=3,
        optimizer_class: Any = Adam,
        lr=3e-4,
        model_loss=None,
        temp_behavior="bce_exp",
        huber_delta=1.0,
        scaler='null',
    ):
        self.transition_model = model
        self.mu = actor
        self.Q = critic
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.temp_model_loss = temp_behavior
        self.model_optimizer = optimizer_class(list(self.transition_model.parameters()), lr=lr)
        self.gamma = gamma
        self.T = T
        self.scaler = scaler_map[scaler](model.nx)

    def update(self, data):
        device = next(self.transition_model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        if hasattr(self.transition_model, "update_input_stats"):
            self.transition_model.update_input_stats(observations, actions)
        # dones = data.dones.to(device=device, dtype=torch.float32)

        pred_next_observations, _, _ = self.transition_model(observations, actions)
        model_value = self.Q(pred_next_observations, self.mu(pred_next_observations))
        target_value = self.Q(next_observations, self.mu(next_observations)).detach()

        if self.temp_model_loss == "bce_exp":
            target_q_prob = torch.exp(-target_value)
            model_q_prob = torch.exp(-model_value)
            dynamics_loss = torch.nn.functional.binary_cross_entropy(model_q_prob, target_q_prob)
        elif self.temp_model_loss == "mse":
            target_q_prob = target_value
            model_q_prob = model_value
            dynamics_loss = torch.nn.functional.mse_loss(model_q_prob, target_q_prob)
        elif self.temp_model_loss == "vaml":
            target_q_prob = target_value
            model_q_prob = model_value
            dynamics_loss = torch.nn.functional.mse_loss(model_q_prob, target_q_prob)
        else:
            raise ValueError(f"Unknown temp_model_loss={self.temp_model_loss}. Expected one of 'bce_exp', 'mse', 'vaml'.")

        loss = dynamics_loss

        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return dynamics_loss, torch.tensor(0.0)


class EnsembleTrainer():
    def __init__(
        self,
        model,
        optimizer_class: Any = Adam,
        lr=3e-4,
        model_loss=None,
        huber_delta=1.0,
        scaler='null',
    ):
        self.model = model
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)
        self.scaler = scaler_map[scaler](model.models[0].nx)

    def update(self, data):
        device = next(self.model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        if hasattr(self.model, "update_input_stats"):
            self.model.update_input_stats(observations, actions)
        elif hasattr(self.model, "models"):
            for member_model in self.model.models:
                if hasattr(member_model, "update_input_stats"):
                    member_model.update_input_stats(observations, actions)

        dynamics_losses = []
        reward_losses = []
        for member_model in self.model.models:
            if hasattr(member_model, "predict_distribution"):
                pred_next_observations, pred_rewards, pred_logvar = member_model.predict_distribution(observations, actions)
                pred_dynamics = pred_next_observations - observations
                target_dynamics = next_observations - observations
                scaled_pred_dynamics = self.scaler.scale(pred_dynamics)
                scaled_target_dynamics = self.scaler.scale(target_dynamics)
                dynamics_loss = gaussian_nll_diag(scaled_pred_dynamics, scaled_target_dynamics, pred_logvar)
            else:
                member_outputs = member_model(observations, actions)
                if len(member_outputs) == 3:
                    pred_next_observations, pred_rewards, _ = member_outputs
                elif len(member_outputs) == 2:
                    pred_next_observations, pred_rewards = member_outputs
                else:
                    raise ValueError(
                        f"Expected member model to return 2 or 3 outputs, got {len(member_outputs)}."
                    )
                pred_dynamics = pred_next_observations - observations
                target_dynamics = next_observations - observations
                scaled_pred_dynamics = self.scaler.scale(pred_dynamics)
                scaled_target_dynamics = self.scaler.scale(target_dynamics)
                dynamics_loss = self.model_loss(scaled_pred_dynamics, scaled_target_dynamics)
            reward_loss = self.model_loss(pred_rewards, rewards)

            dynamics_losses.append(dynamics_loss)
            reward_losses.append(reward_loss)

        mean_dynamics_loss = torch.stack(dynamics_losses).mean()
        mean_reward_loss = torch.stack(reward_losses).mean()
        loss = mean_dynamics_loss + mean_reward_loss

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return mean_dynamics_loss, mean_reward_loss
