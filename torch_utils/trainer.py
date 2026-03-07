import torch.nn as nn
from torch.optim import Adam

class Trainer():
    def __init__(self, model, optimizer_class=Adam, lr=3e-4, model_loss=nn.MSELoss()):
        self.model = model
        self.model_loss = model_loss
        self.model_optimizer = optimizer_class(list(self.model.parameters()), lr=lr)

    def update(self, data):
        pred_next_observations, pred_rewards = self.model(data.observations, data.actions)
        dynamics_loss = self.model_loss(pred_next_observations, data.next_observations)
        reward_loss = self.model_loss(pred_rewards, data.rewards)
        loss = dynamics_loss + reward_loss
        
        # Optimize the model
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return dynamics_loss, reward_loss