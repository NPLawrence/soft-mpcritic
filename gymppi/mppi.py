import numpy as np
import torch
from torch.distributions import MultivariateNormal

# reference:
# https://doi.org/10.1109/ICRA.2017.7989202 "Information theoretic MPC for model-based reinforcement learning"
# https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py



class MPPI():
    def __init__(self, env, rollout_envs=None, gamma=0.99, transition_model=None, Q=None, mu=None, B=1, T=10, K=100, mean=None, cov=None, lambda_=1.0, u_init=None, control_mode="default"):
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

        if self.control_mode not in {"default", "mu", "integrator", "traj_integrator", "mean_residual", "warmstart_residual"}:
            raise ValueError(f"Invalid control_mode={self.control_mode}. Expected one of 'default', 'mu', 'integrator', 'traj_integrator', 'mean_residual', 'warmstart_residual'.")

        self.B = B # batch size
        self.K = K # number of trajectory samples
        self.T = T # length of trajectories/horizon
        self.discounting = self.gamma**torch.arange(self.T+1)

        assert self.transition_model or self.rollout_envs # use either (f,l) or rollout_envs to simulate rollouts
        if self.rollout_envs:
            assert self.rollout_envs.num_envs == self.K # one rollout_env for each sample

        self.lambda_ = lambda_ # free energy scale/parameter
        self.mean = torch.zeros(self.nu, dtype=self.dtype) if (mean is None) else mean.to(dtype=self.dtype) # mean of action noise distribution
        self.cov = torch.diag(torch.ones(self.nu, dtype=self.dtype)) if (cov is None) else cov.to(dtype=self.dtype) # covariance matrix of action noise distribution
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

    @torch.no_grad()
    def make_step(self, observation, mode="action"):
        """return action for given observation"""
        obs_tensor = torch.tensor(observation, dtype=self.dtype)
        if obs_tensor.ndim == 2:
            self.observation = obs_tensor.view(self.B, self.ns)
        elif obs_tensor.ndim == 3 and obs_tensor.shape[1] == 1:
            self.observation = obs_tensor[:, 0].view(self.B, self.ns)
        else:
            raise ValueError(f"Expected observation shape [B,ns] or [B,1,ns], got {tuple(obs_tensor.shape)}")

        # initialize action sequence with previous solution
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

        # Sample all noise at once: [B, K, T+1, nu]
        self.noise = self.noise_dist.rsample((self.B, self.K, self.T + 1))

        if self.transition_model is not None:
            # --- Fully batched path: vectorise over B × K ---
            rollout_cost = self._compute_rollout_costs_batched()  # [B, K]

            if self.control_mode in {"mean_residual", "warmstart_residual"}:
                perturbation_base = self.delta_U  # [B, 1, T+1, nu]
            else:
                perturbation_base = self.U        # [B, 1, T+1, nu]

            # action_cost: [B, K, T+1, nu]; perturbation_cost: [B, K]
            action_cost = self.lambda_ * self.noise @ self.inv_cov
            perturbation_cost = torch.sum(
                self.discounting.view(1, 1, -1, 1) * perturbation_base * action_cost,
                dim=(2, 3),
            )  # [B, K]

            rep_noise = self.noise                                          # [B, K, T+1, nu]

            cost_total = rollout_cost + perturbation_cost                  # [B, K]
            beta = torch.min(cost_total, dim=1, keepdim=True).values       # [B, 1]
            cost_total_non_zero = torch.exp(-(1.0 / self.lambda_) * (cost_total - beta))  # [B, K]

            if mode == 'value':
                logmeanexp = torch.log(torch.mean(cost_total_non_zero, dim=1))           # [B]
                term1 = -self.lambda_ * (-beta.squeeze(1) / self.lambda_ + logmeanexp)
                term2 = self.lambda_ / 2 * torch.sum(
                    self.discounting.view(1, 1, -1, 1) * perturbation_base * (perturbation_base @ self.inv_cov),
                    dim=(1, 2, 3),
                )  # [B]
                self.value = -(term1 + term2).unsqueeze(1)  # [B, 1]

            eta = torch.sum(cost_total_non_zero, dim=1, keepdim=True)  # [B, 1]
            omega = cost_total_non_zero / eta                           # [B, K]
            perturbations = torch.einsum('bk,bktu->btu', omega, rep_noise)  # [B, T+1, nu]
            self.U[:, 0] = self.U[:, 0] + perturbations

        else:
            # --- Sequential fallback for rollout_envs (requires B=1 at K envs for exact equivalence) ---
            for b in range(self.B):
                self.b = b
                self.rollout_observation = self.observation[b]
                self._sync_envs(self.rollout_observation.numpy())
                rollout_cost_b = self._compute_rollout_costs_single(self.rollout_observation)

                if self.control_mode in {"mean_residual", "warmstart_residual"}:
                    perturbation_base_b = self.delta_U[b]
                else:
                    perturbation_base_b = self.U[b]

                action_cost_b = self.lambda_ * self.noise[b] @ self.inv_cov
                perturbation_cost_b = torch.sum(
                    self.discounting.view(1, -1, 1) * perturbation_base_b * action_cost_b, dim=(1, 2)
                )
                rep_noise_b = self.noise[b]

                cost_total_b = rollout_cost_b + perturbation_cost_b
                beta_b = torch.min(cost_total_b)
                cost_total_non_zero_b = torch.exp(-(1.0 / self.lambda_) * (cost_total_b - beta_b))

                if mode == 'value':
                    logmeanexp = torch.log(torch.mean(cost_total_non_zero_b))
                    term1 = -self.lambda_ * (-beta_b / self.lambda_ + logmeanexp)
                    term2 = self.lambda_ / 2 * torch.sum(
                        self.discounting.view(1, -1, 1) * perturbation_base_b * (perturbation_base_b @ self.inv_cov),
                        dim=(1, 2),
                    )
                    self.value[b] = -(term1 + term2)

                eta_b = torch.sum(cost_total_non_zero_b)
                omega_b = cost_total_non_zero_b / eta_b
                perturbations_b = torch.sum(omega_b.view(-1, 1, 1) * rep_noise_b, dim=0)
                self.U[b] = self.U[b] + perturbations_b

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
    def get_value(self, observation, U_init=None):
        self.reset(U_init)
        self.make_step(observation, mode='value')
        return self.value, self.U

    def _compute_rollout_costs_batched(self):
        """Fully-batched rollout over B × K; only valid with transition_model.

        Returns rollout costs shaped [B, K].
        """
        B, K, T, ns, nu = self.B, self.K, self.T, self.ns, self.nu

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
        self.rollout_actions = torch.stack(actions, dim=1)        # [B*K, steps, nu]
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
        self.noise[:, :, t, :] = residual - base_expanded           # update in-place (clamping propagation)

        return residual.reshape(B * K, nu)

    def _compute_rollout_costs_single(self, rollout_observation):
        """compute cost/reward for K trajectories (sequential; used by rollout_envs path)"""
        rollout_cost = torch.zeros(self.K, dtype=self.dtype)

        observation = rollout_observation.repeat(self.K, 1)
        observations = [observation]
        actions = []

        action = self.last_action[self.b, 0].view(1, -1).repeat(self.K, 1)
        if self.control_mode == "mean_residual":
            mean_action = self.mean_U[self.b, 0, 0].view(1, -1).repeat(self.K, 1)
        elif self.control_mode == "warmstart_residual":
            baseline_traj = self.traj_baseline_U[self.b, 0]
        for t in range(self.T):
            residual = self._get_perturbed_action_single(t)
            
            if self.control_mode == "mu":
                if self.mu is None:
                    raise ValueError("control_mode='mu' requires a non-None mu policy/controller.")
                action = self.mu(observation) + residual
            elif self.control_mode == "integrator":
                action = action + residual
            elif self.control_mode == "traj_integrator":
                action = self.last_U[self.b, 0, [t]] + residual
            elif self.control_mode == "mean_residual":
                action = mean_action + residual
            elif self.control_mode == "warmstart_residual":
                action = baseline_traj[t].view(1, -1).repeat(self.K, 1) + residual
            else:
                action = residual
            
            # Ensure action is within bounds
            action = self._bound_action(action)


            next_observation, l = self._step_rollout(observation, action)
            rollout_cost += self.discounting[t] * l

            observation = next_observation

            actions.append(action)
            observations.append(observation)

        if self.Q is not None:
            residual = self._get_perturbed_action_single(self.T)
            
            if self.control_mode == "mu":
                if self.mu is None:
                    raise ValueError("control_mode='mu' requires a non-None mu policy/controller.")
                action = self.mu(observation) + residual
            elif self.control_mode == "integrator":
                action = action + residual
            elif self.control_mode == "traj_integrator":
                action = self.last_U[self.b, 0, [self.T]] + residual
            elif self.control_mode == "mean_residual":
                action = mean_action + residual
            elif self.control_mode == "warmstart_residual":
                action = baseline_traj[self.T].view(1, -1).repeat(self.K, 1) + residual
            else:
                action = residual
            
            # Ensure action is within bounds
            action = self._bound_action(action)
            
            rollout_cost += self.discounting[self.T] * -self.Q(observation, action).flatten()

            next_observation, _ = self._step_rollout(observation, action)
            observation = next_observation

            actions.append(action)
            observations.append(observation)

        # Actions is K x T x nu or K x T+1 x nu if self.Q
        # Observations is K x T+1 x nx or K x T+2 x ns if self.Q
        self.rollout_actions = torch.stack(actions, dim=-2)
        self.rollout_observations = torch.stack(observations, dim=-2)

        return rollout_cost
    
    def _get_perturbed_action_single(self, t):
        """Return residual [K, nu] for the current self.b batch element (rollout_envs path)."""
        if self.control_mode in {"mean_residual", "warmstart_residual"}:
            base = self.delta_U[self.b, 0, t]  # [nu] → broadcast below
        else:
            base = self.U[self.b, 0, t]        # [nu]

        base_k = base.unsqueeze(0).expand(self.K, self.nu)    # [K, nu]
        residual = base_k + self.noise[self.b, :, t]          # [K, nu]
        self.noise[self.b, :, t] = residual - base_k
        return residual

    def _bound_action(self, action):
        """bound action within action space"""
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def _step_rollout(self, observation, action):
        """step the trajectory forward with the appropriate dynamics and cost models or environments"""
        if self.transition_model is not None:
            next_observations, rewards = self.transition_model(observation, action)
            return next_observations.to(self.dtype), -rewards.flatten().to(self.dtype)
        else:
            next_observation, rewards, terminations, truncations, infos = self.rollout_envs.step(action.numpy())
            return torch.from_numpy(next_observation).view(self.K, self.ns).to(self.dtype), -torch.from_numpy(rewards).to(self.dtype)

    def _sync_envs(self, observation):
        """align the mppi environments"""
        if self.rollout_envs:
            self.rollout_envs.reset(options={'observation':observation,
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

if __name__ == '__main__':
    import gymnasium as gym
    from env import BaseEnvWrapper, ClassicMPPIWrapper, MujocoMPPIWrapper
    env_id = "HalfCheetah-v5"
    if any(s in env_id for s in ['Swimmer', 'Hopper', 'Walker', 'Cheetah', 'Ant', 'Humanoid']):
        env_kwargs = {'exclude_current_positions_from_observation': False}
    else:
        env_kwargs = {}

    B = 1
    K = 100
    T = 30

    control_mode = "warmstart_residual"

    env = BaseEnvWrapper(gym.make(env_id, render_mode='human', **env_kwargs))
    ns = env.observation_space.shape[0]
    rollout_envs = gym.make_vec(env_id, num_envs=K, vectorization_mode="sync", wrappers=[MujocoMPPIWrapper], **env_kwargs)

    cov = 0.1*torch.diag(torch.ones(np.prod(env.action_space.shape)))
    mppi = MPPI(env, rollout_envs=rollout_envs, K=K, T=T, B=B, cov=cov, lambda_=0.1, control_mode=control_mode)

    obs, _ = env.reset(seed=0)
    env.render()
    
    cumulative_reward = 0.0
    for _ in range(100):
        batch = np.repeat(obs[None, :], B, axis=0)
        belief = batch + 0.*np.random.randn(B, ns)
        action = mppi.make_step(belief)
        # value = mppi.get_value(belief)
        # action = env.action_space.sample().reshape([1,1,-1])
            
        obs, reward, _, _, info = env.step(action[0,0])
        cumulative_reward += reward
        print(reward)
        env.render()

    print("Cumulative reward:", cumulative_reward)
    env.close()
    rollout_envs.close()
