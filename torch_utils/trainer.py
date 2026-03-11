import torch
import torch.nn as nn
from torch.optim import Adam

class Trainer():
    def __init__(
        self,
        model,
        optimizer_class=Adam,
        lr=3e-4,
        model_loss=None,
        predict_delta=True,
        huber_delta=1.0,
    ):
        self.model = model
        self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)
        self.predict_delta = predict_delta

    def update(self, data):
        device = next(self.model.parameters()).device
        observations = data.observations.to(device=device, dtype=torch.float32)
        actions = data.actions.to(device=device, dtype=torch.float32)
        next_observations = data.next_observations.to(device=device, dtype=torch.float32)
        rewards = data.rewards.to(device=device, dtype=torch.float32)

        pred_next_observations, pred_rewards = self.model(observations, actions)

        if self.predict_delta:
            pred_dynamics = pred_next_observations - observations
            target_dynamics = next_observations - observations
        else:
            pred_dynamics = pred_next_observations
            target_dynamics = next_observations

        dynamics_loss = self.model_loss(pred_dynamics, target_dynamics)
        reward_loss = self.model_loss(pred_rewards, rewards)
        loss = dynamics_loss + reward_loss
        
        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return dynamics_loss, reward_loss
    

# class Trainer_ValueAligned():
#     def __init__(
#         self,
#         model,
#         actor,
#         critic,
#         gamma=0.99,
#         T=3,
#         optimizer_class=Adam,
#         lr=3e-4,
#         model_loss=None,
#         predict_delta=True,
#         huber_delta=1.0,
#     ):
#         self.model = model
#         self.mu = actor
#         self.Q = critic
#         self.model_loss = nn.SmoothL1Loss(beta=huber_delta) if model_loss is None else model_loss
#         self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)
#         self.gamma = gamma
#         self.T = T

#     def update(self, data):
#         device = next(self.model.parameters()).device
#         observations = data.observations.to(device=device, dtype=torch.float32)
#         actions = data.actions.to(device=device, dtype=torch.float32)
#         next_observations = data.next_observations.to(device=device, dtype=torch.float32)
#         rewards = data.rewards.to(device=device, dtype=torch.float32)

#         pred_next_observations, pred_rewards = self.model(observations, actions)

#         # compute value-aligned model loss
#         obs = observations.copy()
#         # rollout_cost =
#         for t in range(self.T):
#             pred_actions = self.mu(pred_next_observations)
#             pred_Q_values = self.Q(pred_next_observations, pred_actions)
#             target_Q_values = rewards + self.gamma * self.Q(next_observations, pred_actions).detach()
#         with torch.no_grad():

#         dynamics_loss = self.model_loss(pred_dynamics, target_dynamics)
#         reward_loss = self.model_loss(pred_rewards, rewards)
#         loss = dynamics_loss + reward_loss
        
#         # Optimize the model
#         self.model_optimizer.zero_grad()
#         loss.backward()
#         self.model_optimizer.step()

#         return dynamics_loss, reward_loss