import torch
import torch.nn as nn
from torch.optim.adam import Adam
from typing import Any

class Trainer():
    def __init__(
        self,
        model,
        optimizer_class: Any = Adam,
        lr=3e-4,
        model_loss=None,
        huber_delta=1.0,
    ):
        self.model = model
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)

    def update(self, data):
        device = next(self.model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        pred_next_observations, pred_rewards, pred_terminations = self.model(observations, actions)

        pred_dynamics = pred_next_observations - observations
        target_dynamics = next_observations - observations

        dynamics_loss = self.model_loss(pred_dynamics, target_dynamics)
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
    ):
        self.transition_model = model
        self.mu = actor
        self.Q = critic
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.temp_model_loss = temp_behavior
        self.model_optimizer = optimizer_class(list(self.transition_model.parameters()), lr=lr)
        self.gamma = gamma
        self.T = T

    def update(self, data):
        device = next(self.transition_model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
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
    ):
        self.model = model
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)

    def update(self, data):
        device = next(self.model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        dynamics_losses = []
        reward_losses = []
        for member_model in self.model.models:
            pred_next_observations, pred_rewards, pred_terminations = member_model(observations, actions)

            pred_dynamics = pred_next_observations - observations
            target_dynamics = next_observations - observations

            dynamics_loss = self.model_loss(pred_dynamics, target_dynamics)
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
