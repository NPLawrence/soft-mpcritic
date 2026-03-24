import numpy as np
import torch
from torch.distributions import MultivariateNormal

# reference:
# https://doi.org/10.1109/ICRA.2017.7989202 "Information theoretic MPC for model-based reinforcement learning"
# https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py



class MPPI():
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

        # Sample all noise at once: [B, K, T+1, nu]
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

        if self.control_mode in {"mean_residual", "warmstart_residual"}:
            perturbation_base = self.delta_U  # [B, 1, T+1, nu]
        else:
            perturbation_base = self.U        # [B, 1, T+1, nu]

        # action_cost: [B, K, T+1, nu]; perturbation_cost: [B, K]
        action_cost = self.lambda_ * self.noise @ self.inv_cov
        perturbation_cost = torch.sum(
            self.discounting.view(1, 1, -1, 1) * perturbation_base * action_cost,
            # perturbation_base * action_cost,
            dim=(2, 3),
        )  # [B, K]

        rep_noise = self.noise                                          # [B, K, T+1, nu]

        cost_total = rollout_cost + perturbation_cost                  # [B, K]
        beta = torch.min(cost_total, dim=1, keepdim=True).values       # [B, 1]
        cost_total_non_zero = torch.exp(-(1.0 / self.lambda_) * (cost_total - beta))  # [B, K]

        if mode == 'value':
            logsumexp = torch.log(torch.sum(cost_total_non_zero, dim=1))           # [B]
            term1 = -self.lambda_ * (-beta.squeeze(1) / self.lambda_ + logsumexp)
            term2 = self.lambda_ / 2 * torch.sum(
                self.discounting.view(1, 1, -1, 1) * perturbation_base * (perturbation_base @ self.inv_cov),
                # perturbation_base * (perturbation_base @ self.inv_cov),
                dim=(1, 2, 3),
            )  # [B]
            self.value = -(term1 + term2).unsqueeze(1)  # [B, 1]

        eta = torch.sum(cost_total_non_zero, dim=1, keepdim=True)  # [B, 1]
        omega = cost_total_non_zero / eta                           # [B, K]
        perturbations = torch.einsum('bk,bktu->btu', omega, rep_noise)  # [B, T+1, nu]
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
        # with clamping called after _get_perturbed_action_batched, this isn't doing anything
        self.noise[:, :, t, :] = residual - base_expanded           # update in-place (clamping propagation)

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

if __name__ == '__main__':
    import gymnasium as gym
    from env import BaseEnvWrapper, ClassicMPPIWrapper, MujocoMPPIWrapper
    env_id = "HalfCheetah-v5"
    if any(s in env_id for s in ['Swimmer', 'Hopper', 'Walker', 'Cheetah', 'Humanoid']):
        env_kwargs = {'exclude_current_positions_from_observation': False}
    elif 'Ant' in env_id:
        env_kwargs = {'exclude_current_positions_from_observation': False, 'include_cfrc_ext_in_observation': False, 'contact_cost_weight': 0.}
    else:
        env_kwargs = {}

    B = 1
    K = 100
    T = 30

    control_mode = "default"

    env = BaseEnvWrapper(gym.make(env_id, render_mode='human', **env_kwargs))
    ns = env.observation_space.shape[0]
    rollout_envs = [gym.make_vec(env_id, num_envs=K, vectorization_mode="sync", wrappers=[MujocoMPPIWrapper], **env_kwargs) for _ in range(B)]

    cov = 0.1*torch.diag(torch.ones(np.prod(env.action_space.shape)))
    mppi = MPPI(env, rollout_envs=rollout_envs, K=K, T=T, B=B, cov=cov, lambda_=0.1, control_mode=control_mode)

    obs, _ = env.reset(seed=0)
    env.render()
    
    cumulative_reward = 0.0
    for _ in range(1000):
        batch = np.repeat(obs[None, :], B, axis=0)
        belief = batch + 0.*np.random.randn(B, ns)
        action = mppi.make_step(belief)
        # value = mppi.get_value(belief)
        # action = env.action_space.sample().reshape([1,1,-1])
            
        next_obs, reward, _, _, info = env.step(action[0,0])

        s, a, s_ = map(torch.from_numpy, [obs, action, next_obs])
        r = env.get_torch_reward(s, a ,s_)
        rew = torch.tensor([reward], dtype=torch.float32)

        print(torch.isclose(r, rew))

        cumulative_reward += reward
        # print(reward)
        env.render()

        obs = next_obs

    print("Cumulative reward:", cumulative_reward)
    env.close()
    [env.close() for env in rollout_envs]
