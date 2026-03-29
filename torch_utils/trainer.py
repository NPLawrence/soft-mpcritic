import torch
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F
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

class MPPITrainer:
    def __init__(
        self,
        args,
        rb,
        qf1, qf1_target,
        qf2, qf2_target,
        q_optimizer,
        target_mppi1, target_mppi2,
        target_horizon,
        mppi_target_iterations,
        target_network_frequency,
        transition_trainer,
    ):
        self.args = args
        self.rb = rb
        self.qf1, self.qf1_target = qf1, qf1_target
        self.qf2, self.qf2_target = qf2, qf2_target
        self.q_optimizer = q_optimizer
        self.target_mppi1, self.target_mppi2 = target_mppi1, target_mppi2
        self.target_horizon = target_horizon
        self.mppi_target_iterations = mppi_target_iterations
        self.target_network_frequency = target_network_frequency
        self.transition_trainer = transition_trainer
        self.training_pattern = args.training_pattern
        self.last_done_global_step = args.learning_starts

        self.last_qf1_a_values = torch.zeros(self.args.batch_size, 1)
        self.last_qf1_loss = torch.tensor([0.])
        self.last_qf_loss = torch.tensor([0.])
        self.last_actor_loss = torch.tensor([0.])
        self.last_dynamics_loss = torch.tensor([0.])
        self.last_reward_loss = torch.tensor([0.])
        self.last_qf2_a_values = torch.zeros(self.args.batch_size, 1)
        self.last_qf2_loss = torch.tensor([0.])

    def _sample(self):
        data, batch_inds, env_indices = self.rb.sample(self.args.batch_size)
        return data, batch_inds, env_indices

    def _update_Q(self, global_step):
        args = self.args

        data, batch_inds, env_indices = self._sample()
        with torch.no_grad():
            if args.mppi and args.mppi_targets:
                mppi_next_observations = data.next_observations.reshape(args.batch_size, -1)

                U_init = data.Us[:, 1:self.target_horizon + 2] if args.mppi_target_warmstart else None

                qf1_next_target, next_Us1 = self.target_mppi1.get_value(
                    mppi_next_observations, U_init=U_init, num_iters=self.mppi_target_iterations
                )
                if args.double_Q:
                    qf2_next_target, next_Us2 = self.target_mppi2.get_value(
                        mppi_next_observations, U_init=U_init, num_iters=self.mppi_target_iterations
                    )

                if args.double_Q:
                    qf_next_target = torch.concat([qf1_next_target, qf2_next_target], dim=1)
                    next_Us_target = torch.concat([next_Us1, next_Us2], dim=1)
                    min_qf_next_target, indices = torch.min(qf_next_target, dim=1)
                    indices_expanded = indices[:, None, None, None].expand(
                        -1, 1, next_Us_target.shape[2], next_Us_target.shape[3]
                    )
                    next_Us = next_Us_target.gather(1, indices_expanded)
                else:
                    min_qf_next_target, next_Us = qf1_next_target, next_Us1

                # self.rb.Us[batch_inds, env_indices, 1:self.target_horizon + 1] = next_Us[:, 0, :self.target_horizon]
                if self.target_horizon < args.horizon:
                    self.rb.Us[batch_inds, env_indices, 1:self.target_horizon+2] = next_Us[:,0,:self.target_horizon+1] # copy over solution offset by 1 step, including time step
                elif self.target_horizon == args.horizon:
                    self.rb.Us[batch_inds, env_indices, 1:self.target_horizon+1] = next_Us[:,0,:self.target_horizon] # copy over solution offset by 1 step
                else:
                    raise NotImplementedError(f"Not implemented target_horizon > horizon. Got target_horizon={self.target_horizon} and horizon={args.horizon}.")
                next_q_value = (
                    data.rewards.flatten()
                    + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)
                )

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf1_loss = self._q_loss(qf1_a_values, next_q_value)

        if args.double_Q:
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf2_loss = self._q_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
        else:
            qf2_a_values = torch.zeros_like(qf1_a_values)
            qf2_loss = torch.zeros_like(qf1_loss)
            qf_loss = qf1_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % self.target_network_frequency == 0:
            self._soft_update(self.qf1, self.qf1_target)
            if args.double_Q:
                self._soft_update(self.qf2, self.qf2_target)

        return qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss

    def _update_f(self):
        args = self.args
        if args.mppi and not args.env_in_mppi and args.horizon > 0:
            for _ in range(args.transition_utd):
                data, _, _ = self._sample()
                dynamics_loss, reward_loss = self.transition_trainer.update(data)
        else:
            dynamics_loss = torch.tensor([0.])
            reward_loss = torch.tensor([0.])

        return dynamics_loss, reward_loss

    def update(self, global_step, infos):
        args = self.args
        episode_ended = "final_info" in infos
        is_first_update = global_step == self.args.learning_starts + 1

        if 'episodic' in args.training_pattern:
            num_updates = global_step - self.last_done_global_step
            if self.training_pattern == 'only_model_episodic':
                if (episode_ended or is_first_update):
                    for _ in range(num_updates):
                        dynamics_loss, reward_loss = self._update_f()
                else:
                    dynamics_loss, reward_loss = self.last_dynamics_loss, self.last_reward_loss
                qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = self._update_Q(global_step)
            elif self.training_pattern == 'episodic_model_first':
                if (episode_ended or is_first_update):
                    for _ in range(num_updates):
                        dynamics_loss, reward_loss = self._update_f()
                    for k in range(num_updates):
                        effective_global_step = self.last_done_global_step + k
                        qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = self._update_Q(effective_global_step)
                else:
                    qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = (
                        self.last_qf1_a_values, self.last_qf1_loss, self.last_qf_loss,
                        self.last_qf2_a_values, self.last_qf2_loss
                    )
                    dynamics_loss, reward_loss = self.last_dynamics_loss, self.last_reward_loss
            elif self.training_pattern == 'episodic':
                if (episode_ended or is_first_update):
                    for k in range(num_updates):
                        effective_global_step = self.last_done_global_step + k
                        qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = self._update_Q(effective_global_step)
                        dynamics_loss, reward_loss = self._update_f()
                else:
                    qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = (
                        self.last_qf1_a_values, self.last_qf1_loss, self.last_qf_loss,
                        self.last_qf2_a_values, self.last_qf2_loss
                    )
                    dynamics_loss, reward_loss = self.last_dynamics_loss, self.last_reward_loss
        elif self.training_pattern == 'online':
            qf1_a_values, qf1_loss, qf_loss, qf2_a_values, qf2_loss = self._update_Q(global_step)
            dynamics_loss, reward_loss = self._update_f()

        if episode_ended:
            self.last_done_global_step = global_step

        actor_loss = torch.tensor([0.])
        self._update_last(qf1_a_values, qf1_loss, qf_loss, actor_loss, dynamics_loss, reward_loss, qf2_a_values, qf2_loss)

        return qf1_a_values, qf1_loss, qf_loss, actor_loss, dynamics_loss, reward_loss, qf2_a_values, qf2_loss

    def _q_loss(self, predicted, target):
        if self.args.use_huber_loss:
            return F.smooth_l1_loss(predicted, target, beta=self.args.huber_delta)
        return F.mse_loss(predicted, target)

    def _soft_update(self, online, target):
        for param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(
                self.args.tau * param.data + (1 - self.args.tau) * target_param.data
            )
    
    def _update_last(self, qf1_a_values, qf1_loss, qf_loss, actor_loss, dynamics_loss, reward_loss, qf2_a_values, qf2_loss):
        self.last_qf1_a_values = qf1_a_values
        self.last_qf1_loss = qf1_loss
        self.last_qf_loss = qf_loss
        self.last_actor_loss = actor_loss
        self.last_dynamics_loss = dynamics_loss
        self.last_reward_loss = reward_loss
        self.last_qf2_a_values = qf2_a_values
        self.last_qf2_loss = qf2_loss