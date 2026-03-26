import numpy as np
import torch
from torch.distributions import MultivariateNormal

# Uniform-prior MPPI: prior p = U(u_min, u_max) per dimension,
# controlled distribution q = N(U, Σ) (Gaussian, same as standard MPPI).
#
# IS correction: λ log(q(v)/p(v)) — constant-in-p cancels; non-constant part:
#   perturbation_cost = -λ/2 * Σ_t γ^t ε_t^T Σ⁻¹ ε_t
# There is no U-dependent term2 (unlike Gaussian MPPI) so the value is just term1.
#
# reference:
# https://doi.org/10.1109/ICRA.2017.7989202 "Information theoretic MPC for model-based reinforcement learning"
# https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py


class UniformMPPI():
    def __init__(self, env, rollout_envs=None, gamma=0.99, transition_model=None, Q=None, mu=None, B=1, T=10, K=100, mean=None, cov=None, lambda_=1.0, u_init=None, control_mode="default", ensemble_rollout_mode="trajectory"):
        self.env = env
        self.rollout_envs = rollout_envs
        self.env_dtype = self.env.observation_space.dtype
        self.dtype = torch.float32

        self.ns = np.prod(self.env.observation_space.shape)
        self.nu = np.prod(self.env.action_space.shape)
        self.u_min = torch.from_numpy(self.env.action_space.low).to(dtype=self.dtype)
        self.u_max = torch.from_numpy(self.env.action_space.high).to(dtype=self.dtype)

        self.gamma = gamma
        self.transition_model = transition_model # model of p(s',r|s,a)
        self.Q = Q # terminal value function (goal is to minimize u^* = argmin -Q(s,a))
        self.mu = mu # controller (used only when control_mode == "mu")

        self.control_mode = control_mode
        self.ensemble_rollout_mode = ensemble_rollout_mode

        if self.control_mode not in {"default", "mu", "integrator", "traj_integrator", "mean_residual", "warmstart_residual"}:
            raise ValueError(f"Invalid control_mode={self.control_mode}. Expected one of 'default', 'mu', 'integrator', 'traj_integrator', 'mean_residual', 'warmstart_residual'.")
        if self.ensemble_rollout_mode not in {"trajectory", "batch"}:
            raise ValueError(
                f"Invalid ensemble_rollout_mode={self.ensemble_rollout_mode}. Expected one of 'trajectory', 'batch'."
            )

        self.B = B # batch size
        self.K = K # number of trajectory samples
        self.T = T # length of trajectories/horizon
        self.discounting = self.gamma**torch.arange(self.T+1)

        assert self.transition_model or self.rollout_envs # use either (f,l) or rollout_envs to simulate rollouts
        if self.rollout_envs:
            assert len(self.rollout_envs) == self.B
            assert all([self.rollout_envs[i].num_envs == self.K for i in range(self.B)]) # one rollout_env for each sample

        self.lambda_ = lambda_ # free energy scale/parameter
        self.mean = torch.zeros(self.nu, dtype=self.dtype) if (mean is None) else mean.to(dtype=self.dtype) # mean of Gaussian controlled distribution
        self.cov = torch.diag(torch.ones(self.nu, dtype=self.dtype)) if (cov is None) else cov.to(dtype=self.dtype) # covariance matrix of Gaussian controlled distribution
        self.inv_cov = torch.inverse(self.cov)
        self.noise_dist = MultivariateNormal(self.mean, covariance_matrix=self.cov)

        self.U = self.noise_dist.sample((self.B, 1, self.T+1,)) # initial action sequence per batch element
        self.noise = torch.zeros(self.B, self.K, self.T+1, self.nu, dtype=self.dtype)
        self.u_init = torch.zeros_like(self.mean) if (u_init is None) else u_init # initialization for last action of trajectory at each step
        
        self.observation = None
        self.info = None

        # sampled results from last command
        self.last_U = self.u_init * torch.ones_like(self.U) # integral trajectory w.r.t. "environment time" where each element is u_t = u_t-1 + \delta_t (index i corresponds to t=i-1)
        self.last_action = self.u_init * torch.ones_like(self.U[:,:,0]) # initial action for integral action w.r.t. "horizon time" where each subsequent element is u_k = u_k-1 + \delta_k (index i corresponds to k=i-1)
        self.rollout_observation = None
        self.rollout_observations = None
        self.rollout_actions = None
        self.value = torch.zeros(self.B, 1)
        self.rollout_model_indices = None

    @torch.no_grad()
    def make_step(self, observation, mode="action", roll=True):
        """return action for given observation"""
        obs_tensor = torch.tensor(observation, dtype=self.dtype)
        if obs_tensor.ndim == 2:
            self.observation = obs_tensor.view(self.B, self.ns)
        elif obs_tensor.ndim == 3 and obs_tensor.shape[1] == 1:
            self.observation = obs_tensor[:, 0].view(self.B, self.ns)
        else:
            raise ValueError(f"Expected observation shape [B,ns] or [B,1,ns], got {tuple(obs_tensor.shape)}")

        # initialize action sequence with previous solution
        if roll:
            self.U = torch.roll(self.U, -1, dims=2)
            self.U[:, :, -1] = self.u_init
        if self.control_mode == "mean_residual":
            self.mean_U = torch.mean(self.U, dim=2, keepdim=True)  # [B, 1, 1, nu]
            self.delta_U = self.U - self.mean_U
        elif self.control_mode == "warmstart_residual":
            self.traj_baseline_U = self.U.clone()  # [B, 1, T+1, nu]
            self.delta_U = torch.zeros_like(self.U)
        else:
            self.delta_U = self.U

        # Sample all noise at once from Gaussian controlled distribution: [B, K, T+1, nu]
        self.noise = self.noise_dist.rsample((self.B, self.K, self.T + 1))

        if self.rollout_envs:
            self._sync_envs()

        if self.transition_model is not None and hasattr(self.transition_model, "ensemble_size") and self.ensemble_rollout_mode == "trajectory":
            if roll or self.rollout_model_indices is None:
                self.rollout_model_indices = torch.randint(
                    self.transition_model.ensemble_size,
                    (self.B * self.K,),
                    dtype=torch.long,
                )
        else:
            self.rollout_model_indices = None

        # --- Fully batched path: vectorise over B × K ---
        rollout_cost = self._compute_rollout_costs_batched()  # [B, K]

        # Prior p = U(u_min, u_max) per dimension (flat over action bounds).
        # Controlled q = N(U, Σ).  IS correction: λ log(q(v)/p(v)).
        # log p is constant, so only the q-dependent part survives:
        #   λ log q(v) = -λ/2 ε^T Σ⁻¹ ε  + const
        # This is fully sample-dependent (no U-dependent term) so it goes
        # entirely into perturbation_cost; no term2 correction is needed.
        #
        # Compare to Gaussian-prior MPPI where log(p/q) = u^T Σ⁻¹ ε - ½ u^T Σ⁻¹ u:
        # here the sign is reversed and the quadratic is in ε not u.
        # A negative cost for on-mean samples down-weights them relative to the
        # flat prior, which is the correct IS correction.

        # perturbation_cost: -λ/2 * Σ_t γ^t ε_t^T Σ⁻¹ ε_t  →  [B, K]
        perturbation_cost = -0.5 * self.lambda_ * torch.sum(
            self.discounting.view(1, 1, -1, 1) * (self.noise @ self.inv_cov) * self.noise,
            dim=(2, 3),
        )  # [B, K]

        cost_total = rollout_cost + perturbation_cost                  # [B, K]
        beta = torch.min(cost_total, dim=1, keepdim=True).values       # [B, 1]
        cost_total_non_zero = torch.exp(-(1.0 / self.lambda_) * (cost_total - beta))  # [B, K]

        if mode == 'value':
            # Free energy: F* = -λ log E_p[exp(-S/λ)], estimated via IS from q.
            # term1 already uses IS-corrected costs (perturbation_cost included);
            # there is no U-dependent term2 (unlike Gaussian-prior MPPI).
            logsumexp = torch.log(torch.sum(cost_total_non_zero, dim=1))           # [B]
            term1 = -self.lambda_ * (-beta.squeeze(1) / self.lambda_ + logsumexp)
            self.value = -term1.unsqueeze(1)  # [B, 1]

        eta = torch.sum(cost_total_non_zero, dim=1, keepdim=True)  # [B, 1]
        omega = cost_total_non_zero / eta                           # [B, K]
        perturbations = torch.einsum('bk,bktu->btu', omega, self.noise)  # [B, T+1, nu]
        self.U[:, 0] = self.U[:, 0] + perturbations

        if self.control_mode == "mu":
            if self.mu is None:
                raise ValueError("control_mode='mu' requires a non-None mu policy/controller.")
            action = self.mu(self.observation).view(self.B, 1, self.nu) + self.U[:, 0, [0]]
        elif self.control_mode == "integrator":
            action = self.last_action + self.U[:, 0, [0]]
        elif self.control_mode == "traj_integrator":
            self.last_U = self.last_U + self.U
            action = self.last_U[:, 0, [0]]
        else:
            action = self.U[:, 0, [0]]  # first action in sequence across batch: [B, 1, nu]

        # Ensure action is within bounds
        action = self._bound_action(action)
        self.last_action = action

        return action.numpy()

    @torch.no_grad()
    def get_action(self, observation, num_iters=1):
        action = self.make_step(observation, mode='action')
        for _ in range(num_iters-1):
            action = self.make_step(observation, mode='action', roll=False)
        return action

    @torch.no_grad()
    def get_value(self, observation, U_init=None, num_iters=1):
        self.reset(U_init)
        self.make_step(observation, mode='value')
        for _ in range(num_iters-1):
            self.make_step(observation, mode='value', roll=False)
        return self.value, self.U

    def _compute_rollout_costs_batched(self):
        """Fully-batched rollout over B × K; only valid with transition_model.

        Returns rollout costs shaped [B, K].
        """
        B, K, T, ns, nu = self.B, self.K, self.T, self.ns, self.nu
        if self.observation is None:
            raise ValueError("Observation must be set before computing rollout costs.")

        # Expand observations: [B, ns] → [B*K, ns]
        observation = (
            self.observation
            .unsqueeze(1).expand(B, K, ns)
            .reshape(B * K, ns)
        )

        rollout_cost = torch.zeros(B * K, dtype=self.dtype)
        observations = [observation]
        actions = []

        # last_action: [B, 1, nu] → [B*K, nu]
        action = (
            self.last_action[:, 0]
            .unsqueeze(1).expand(B, K, nu)
            .reshape(B * K, nu)
        )

        if self.control_mode == "mean_residual":
            # mean_U: [B, 1, 1, nu] → [B*K, nu]
            mean_action = (
                self.mean_U[:, 0, 0]
                .unsqueeze(1).expand(B, K, nu)
                .reshape(B * K, nu)
            )

        for t in range(T):
            residual = self._get_perturbed_action_batched(t)  # [B*K, nu]

            if self.control_mode == "mu":
                if self.mu is None:
                    raise ValueError("control_mode='mu' requires a non-None mu policy/controller.")
                action = self.mu(observation) + residual
            elif self.control_mode == "integrator":
                action = action + residual
            elif self.control_mode == "traj_integrator":
                last_u_t = (
                    self.last_U[:, 0, t]
                    .unsqueeze(1).expand(B, K, nu)
                    .reshape(B * K, nu)
                )
                action = last_u_t + residual
            elif self.control_mode == "mean_residual":
                action = mean_action + residual
            elif self.control_mode == "warmstart_residual":
                baseline_t = (
                    self.traj_baseline_U[:, 0, t]
                    .unsqueeze(1).expand(B, K, nu)
                    .reshape(B * K, nu)
                )
                action = baseline_t + residual
            else:
                action = residual

            action = self._bound_action(action)
            next_observation, l = self._step_rollout(observation, action)
            rollout_cost += self.discounting[t] * l
            observation = next_observation
            actions.append(action)
            observations.append(observation)

        if self.Q is not None:
            residual = self._get_perturbed_action_batched(T)  # [B*K, nu]

            if self.control_mode == "mu":
                if self.mu is None:
                    raise ValueError("control_mode='mu' requires a non-None mu policy/controller.")
                action = self.mu(observation) + residual
            elif self.control_mode == "integrator":
                action = action + residual
            elif self.control_mode == "traj_integrator":
                last_u_T = (
                    self.last_U[:, 0, T]
                    .unsqueeze(1).expand(B, K, nu)
                    .reshape(B * K, nu)
                )
                action = last_u_T + residual
            elif self.control_mode == "mean_residual":
                action = mean_action + residual
            elif self.control_mode == "warmstart_residual":
                baseline_T = (
                    self.traj_baseline_U[:, 0, T]
                    .unsqueeze(1).expand(B, K, nu)
                    .reshape(B * K, nu)
                )
                action = baseline_T + residual
            else:
                action = residual

            action = self._bound_action(action)
            rollout_cost += self.discounting[T] * -self.Q(observation, action).flatten()

            next_observation, _ = self._step_rollout(observation, action)
            observation = next_observation
            actions.append(action)
            observations.append(observation)

        # [B*K, steps, ns/nu] stored for inspection
        self.rollout_actions = (
            torch.stack(actions, dim=1) if actions
            else torch.empty(B * K, 0, nu, dtype=self.dtype)
        )  # [B*K, steps, nu]
        self.rollout_observations = torch.stack(observations, dim=1)  # [B*K, steps+1, ns]

        return rollout_cost.view(B, K)  # [B, K]

    def _get_perturbed_action_batched(self, t):
        """Return residual [B*K, nu]."""
        B, K, nu = self.B, self.K, self.nu
        if self.control_mode in {"mean_residual", "warmstart_residual"}:
            base = self.delta_U[:, 0, t]  # [B, nu]
        else:
            base = self.U[:, 0, t]        # [B, nu]

        base_expanded = base.unsqueeze(1).expand(B, K, nu)          # [B, K, nu]
        residual = base_expanded + self.noise[:, :, t, :]            # [B, K, nu]
        self.noise[:, :, t, :] = residual - base_expanded

        return residual.reshape(B * K, nu)

    def _bound_action(self, action):
        """bound action within action space"""
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def _step_rollout(self, observation, action):
        """step the trajectory forward with the appropriate dynamics and cost models or environments"""
        if self.transition_model is not None:
            if self.rollout_model_indices is not None:
                next_observations, rewards = self.transition_model(
                    observation,
                    action,
                    model_indices=self.rollout_model_indices,
                )
            else:
                next_observations, rewards = self.transition_model(observation, action)
            return next_observations.to(self.dtype), -rewards.flatten().to(self.dtype)
        else:
            if self.rollout_envs is None:
                raise ValueError("rollout_envs must be provided when transition_model is None.")
            action = action.view(self.B, self.K, self.nu)
            next_observations_list = []
            rewards_list = []
            for (rollout_env, act) in zip(self.rollout_envs, action):
                next_obs, rew, _, _, _ = rollout_env.step(act.numpy())
                next_observations_list.append(next_obs)
                rewards_list.append(rew)
            next_observation = np.concatenate(next_observations_list)
            rewards = np.concatenate(rewards_list)
            return torch.from_numpy(next_observation).to(self.dtype), -torch.from_numpy(rewards).to(self.dtype)

    def _sync_envs(self):
        """align the mppi environments"""
        if self.observation is None or self.rollout_envs is None:
            raise ValueError("Observation and rollout_envs must be set before syncing environments.")
        observation = self.observation.numpy()
        for (rollout_env, obs) in zip(self.rollout_envs, observation):
            rollout_env.reset(options={'observation':obs,
                                       'extra':self.env.get_wrapper_attr('_extra')}) # set MPPIEnvs to state

    def reset(self, U_init=None):
        """reinitialize all MPPI computations"""
        self.U = self.noise_dist.sample((self.B, 1, self.T+1,)) if U_init is None else U_init.view((self.B, 1, self.T+1, self.nu))
        self.last_U = self.u_init * torch.ones_like(self.U)
        self.last_action = self.u_init * torch.ones_like(self.U[:,:,0])
        self.noise = torch.zeros(self.B, self.K, self.T+1, self.nu, dtype=self.dtype)
        self.value = torch.zeros(self.B, 1)
        self.observation = None
        self.rollout_observation = None
        self.rollout_observations = None
        self.rollout_actions = None
        self.rollout_model_indices = None

