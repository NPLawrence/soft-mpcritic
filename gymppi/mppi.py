import numpy as np
import torch
from torch.distributions import MultivariateNormal

# reference:
# https://doi.org/10.1109/ICRA.2017.7989202 "Information theoretic MPC for model-based reinforcement learning"
# https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py

class MPPI():
    def __init__(self, env, rollout_envs=None, l=None, f=None, Q=None, T=10, K=100, mu=None, cov=None, lambda_=1.0, u_init=None, U_init=None, u_scale=None):
        self.env = env
        self.rollout_envs = rollout_envs
        self.env_dtype = self.env.observation_space.dtype
        self.dtype = torch.from_numpy(np.array([], dtype=self.env_dtype)).dtype

        self.ns = np.prod(self.env.observation_space.shape)
        self.nu = np.prod(self.env.action_space.shape)
        self.u_min = torch.from_numpy(self.env.action_space.low)
        self.u_max = torch.from_numpy(self.env.action_space.high)

        self.l = l # stage cost
        self.f = f # dynamics
        self.Q = Q # terminal value function (assumes we want u^* = argmin -Q(s,a))

        self.K = K # number of trajectory samples
        self.T = T # length of trajectories/horizon

        assert (self.f and self.l) or self.rollout_envs # use either (f,l) or rollout_envs to simulate rollouts
        if self.rollout_envs:
            assert self.rollout_envs.num_envs == self.K # one rollout_env for each sample

        self.lambda_ = lambda_ # free energy scale/parameter
        self.mu = torch.zeros(self.nu, dtype=self.dtype) if (mu is None) else mu # mean of action noise distribution
        self.cov = torch.diag(torch.ones(self.nu, dtype=self.dtype)) if (cov is None) else cov # covariance matrix of action noise distribution
        self.inv_cov = torch.inverse(self.cov)
        self.noise_dist = MultivariateNormal(self.mu, covariance_matrix=self.cov)

        self.U = self.noise_dist.sample((self.T+1,)) if (U_init is None) else U_init # initial trajectory
        self.u_init = torch.zeros_like(self.mu) if (u_init is None) else u_init # initialization for last action of trajectory at each step
        
        self.observation = None
        self.info = None

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.observations = None
        self.actions = None

    def make_step(self, observation):
        """return action for given observation"""
        self._sync_envs()

        if not torch.is_tensor(observation):
            observation = torch.tensor(observation)
        self.observation = observation.to(dtype=self.dtype)

        # sample \epsilon ~ N(mu, cov)
        noise = self.noise_dist.rsample((self.K, self.T+1))
        # v = u + \epsilon; broadcast U to noise over samples; now it's K x T+1 x nu
        perturbed_action = self.U + noise
        # ensure actions are within action space
        self.perturbed_action = self._bound_action(perturbed_action)
        # reflect action constraints in noise
        noise = self.perturbed_action - self.U


        action_cost = self.lambda_ * noise @ self.inv_cov # (24) inner summation

        rollout_cost = self._compute_rollout_costs(self.perturbed_action)

        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2)) # (24) inner summation
        cost_total = rollout_cost + perturbation_cost # (24) inner sum

        beta = torch.min(cost_total) # "ensure that at least one trajectory has non-zero mass"
        cost_total_non_zero = torch.exp(-self.lambda_ * (cost_total - beta)) # (24) exp
        eta = torch.sum(cost_total_non_zero) # (25)
        omega = (1. / eta) * cost_total_non_zero # (24) normalize for sample weights

        perturbations = torch.sum(omega.view(-1, 1, 1) * noise, dim=0) # (26) summation
        self.U = self.U + perturbations # (26)

        action = self.U[0]
        return action.numpy()
    
    def _compute_rollout_costs(self, perturbed_action):
        """compute cost/reward for trajectory"""
        rollout_cost = torch.zeros(self.K, dtype=self.dtype)

        observation = self.observation.view(1, -1).repeat(self.K, 1)
        observations = [observation]
        actions = []

        with torch.no_grad():
            for t in range(self.T):
                action = perturbed_action[:, t]

                next_observation, l = self._step_rollout(observation, action)
                rollout_cost += l

                observation = next_observation

                actions.append(action)
                observations.append(observation)

            if self.Q is not None:
                action = perturbed_action[:, self.T+1]
                rollout_cost += -self.Q(observation, action).flatten()

                next_observation, _ = self._step_rollout(observation, action)
                observation = next_observation

                actions.append(action)
                observations.append(observation)

        # Actions is K x T+1 x nu
        # Observations is K x T+2 x nx
        self.actions = torch.stack(actions, dim=-2)
        self.observations = torch.stack(observations, dim=-2)

        return rollout_cost
    
    def _bound_action(self, action):
        """bound action within action space"""
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def _step_rollout(self, observation, action):
        """step the trajectory forward with the appropriate dynamics and cost models or environments"""
        if self.f is not None:
            return self.f(observation, action), self.l(observation, action).flatten()
        else:
            next_obsveration, rewards, terminations, truncations, infos = self.rollout_envs.step(action.numpy())
            return torch.from_numpy(next_obsveration).view(-1, self.ns), -torch.from_numpy(rewards)

    def _sync_envs(self):
        """align the """
        if self.rollout_envs:
            state = self.env.env._state # access BaseEnv state
            observation, _ = self.rollout_envs.reset(options={'state':state}) # set MPPIEnvs to state
            self.mppi_observation = observation

    def reset(self, U_init=None):
        """reinitialize all MPPI computations"""
        self.U = self.noise_dist.sample((self.T+1,)) if (U_init is None) else U_init
        self.observation = None
        self.info = None
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.observations = None
        self.actions = None