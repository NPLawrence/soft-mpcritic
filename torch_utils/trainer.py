import torch
import torch.nn as nn
from torch.optim.adam import Adam

class Trainer():
    def __init__(
        self,
        model,
        optimizer_class=Adam,
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

        pred_next_observations, pred_rewards = self.model(observations, actions)

        dynamics_loss = self.model_loss(pred_next_observations, next_observations)
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
        optimizer_class=Adam,
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
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        # compute value-aligned model loss
        obs = observations
        a = actions
        # rollout_cost = torch.zeros_like(rewards)
        for t in range(1):
            if t > 0: # add back if using 1-step bellman target
                a = self.mu(obs)
            obs, rew = self.transition_model(obs, a)
            # rollout_cost += (self.gamma ** t) * rew

        # rollout_cost += (self.gamma ** self.T) * self.Q(obs, self.mu(obs))
        rollout_cost = self.Q(obs, self.mu(obs))

        # bellman_target = rewards + self.gamma*self.Q(next_observations, self.mu(next_observations)).detach()
        # bellman_target = self.Q(observations, self.mu(observations)).detach()
        bellman_target = self.Q(next_observations, self.mu(next_observations)).detach()
        if self.temp_model_loss == "bce_exp":
            target_q_prob = torch.exp(-bellman_target).detach()
            model_q_prob = torch.exp(-rollout_cost)
            dynamics_loss = torch.nn.functional.binary_cross_entropy(model_q_prob, target_q_prob)
        elif self.temp_model_loss == "mse":
            target_q_prob = bellman_target.detach()
            model_q_prob = rollout_cost
            dynamics_loss = torch.nn.functional.mse_loss(model_q_prob, target_q_prob)
        elif self.temp_model_loss == "vaml":
            target_q_prob = bellman_target.detach()
            model_q_prob = rollout_cost
            dynamics_loss = torch.nn.functional.mse_loss(model_q_prob, target_q_prob)

        loss = dynamics_loss

        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return dynamics_loss, torch.tensor(0.0)
