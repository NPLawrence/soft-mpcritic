import numpy as np
import torch
from torch.distributions import MultivariateNormal

# reference:
# https://doi.org/10.1109/ICRA.2017.7989202 "Information theoretic MPC for model-based reinforcement learning"
# https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py



class MPPI():
    def __init__(self, env, rollout_envs=None, gamma=0.99, l=None, f=None, Q=None, mu=None, B=1, P=1, T=10, K=100, mean=None, cov=None, lambda_=1.0, u_init=None):
        self.env = env
        self.rollout_envs = rollout_envs
        self.env_dtype = self.env.observation_space.dtype
        self.dtype = torch.float32

        self.ns = np.prod(self.env.observation_space.shape)
        self.nu = np.prod(self.env.action_space.shape)
        self.u_min = torch.from_numpy(self.env.action_space.low).to(dtype=self.dtype)
        self.u_max = torch.from_numpy(self.env.action_space.high).to(dtype=self.dtype)

        self.gamma = gamma
        self.l = l # stage cost
        self.f = f # dynamics
        self.Q = Q # terminal value function (assumes we want u^* = argmin Q(s,a))
        self.mu = mu # controller

        self.B = B # batch size
        self.P = P # number of particles
        self.K = K # number of trajectory samples
        self.T = T # length of trajectories/horizon
        self.discounting = self.gamma**torch.arange(self.T+1).view(1,-1,1)

        assert (self.f and self.l) or self.rollout_envs # use either (f,l) or rollout_envs to simulate rollouts
        if self.rollout_envs:
            assert self.rollout_envs.num_envs == self.K # one rollout_env for each sample

        self.lambda_ = lambda_ # free energy scale/parameter
        self.mean = torch.zeros(self.nu, dtype=self.dtype) if (mean is None) else mean.to(dtype=self.dtype) # mean of action noise distribution
        self.cov = 1.0*torch.diag(torch.ones(self.nu, dtype=self.dtype)) if (cov is None) else cov.to(dtype=self.dtype) # covariance matrix of action noise distribution
        self.inv_cov = torch.inverse(self.cov)
        self.noise_dist = MultivariateNormal(self.mean, covariance_matrix=self.cov)

        self.U = self.noise_dist.sample((self.B, 1, self.T+1,)) # initial action sequence shared amongst particles
        self.noise = torch.zeros(self.B, self.K, self.T+1, self.nu, dtype=self.dtype)
        self.u_init = torch.zeros_like(self.mean) if (u_init is None) else u_init # initialization for last action of trajectory at each step
        
        self.observation = None
        self.info = None

        # sampled results from last command
        self.rollout_observation = None
        self.rollout_observations = None
        self.rollout_actions = None
        self.value = torch.zeros(self.B, 1)

    @torch.no_grad()
    def make_step(self, observation, mode="action"):
        """return action for given observation"""
        self.observation = torch.tensor(observation, dtype=self.dtype).view(self.B, self.P, self.ns)

        # initialize action sequence with previous solution
        self.U = torch.roll(self.U, -1, dims=2)
        self.U[:,:,-1] = self.u_init

        # calculate batch B actions where each of B is computed across P particles for K noise sequences
        for b in range(self.B):
            self.b = b

            # sample \epsilon ~ N(mean, cov)
            self.noise[b] = self.noise_dist.rsample((self.K, self.T+1))

            # simulate actions + noise over particles to calculate P x K costs
            rollout_costs = []
            for p in range(self.P):
                self.p = p

                self.rollout_observation = self.observation[b,p]
                self._sync_envs(self.rollout_observation.numpy())
                rollout_costs.append(self._compute_rollout_costs(self.rollout_observation))

            rollout_cost = torch.cat(rollout_costs, dim=0)
            action_cost = self.lambda_ * self.noise[b] @ self.inv_cov # (24) inner summation
            perturbation_cost = torch.sum(self.discounting * self.U[b] * action_cost, dim=(1, 2)) # (24) inner summation

            # repeating cost for across P particles because noise is shared
            perturbation_cost = perturbation_cost.repeat(self.P)
            rep_noise = self.noise[b].repeat(self.P,1,1)

            cost_total = rollout_cost + perturbation_cost # (24) inner sum

            beta = torch.min(cost_total) # "ensure that at least one trajectory has non-zero mass"
            cost_total_non_zero = torch.exp(-(1/self.lambda_) * (cost_total - beta)) # (24) exp

            if mode == 'value':
                # negative of "free energy" as approximately max_a Q(s,a)
                logmeanexp = torch.log(torch.mean(cost_total_non_zero)) # compute log sum exp for trajectories
                term1 = -self.lambda_ * (-beta/self.lambda_ + logmeanexp) # energy accounting for added beta/lambda in exp
                term2 = self.lambda_/2 * torch.sum(self.discounting * self.U[b] * (self.U[b] @ self.inv_cov), dim=(1, 2)) # mean term correction (shared among trajectories)
                self.value[b] = -(term1 + term2) # compute F(s)

            eta = torch.sum(cost_total_non_zero) # (25)
            omega = (1. / eta) * cost_total_non_zero # (24) normalize for sample weights

            perturbations = torch.sum(omega.view(-1, 1, 1) * rep_noise, dim=0) # (26) summation
            self.U[b] = self.U[b] + perturbations # (26)

        if self.mu:
            action = self.mu(self.observation) + self.U[:,0,[0]] # first action in sequence actions across batch
        else:
            action = self.U[:,0,[0]] # first action in sequence actions across batch
        
        # Ensure action is within bounds
        action = self._bound_action(action)
        
        return action.numpy()
    
    @torch.no_grad()
    def get_value(self, observation, U_init=None):
        self.reset(U_init)
        self.make_step(observation, mode='value')
        return self.value, self.U

    def _compute_rollout_costs(self, rollout_observation):
        """compute cost/reward for K trajectories"""
        rollout_cost = torch.zeros(self.K, dtype=self.dtype)

        observation = rollout_observation.repeat(self.K, 1)
        observations = [observation]
        actions = []

        for t in range(self.T):
            residual = self._get_perturbed_action(observation, t)
            
            if self.mu:
                action = self.mu(observation) + residual
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
            residual = self._get_perturbed_action(observation, self.T)
            
            if self.mu:
                action = self.mu(observation) + residual
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
    
    def _get_perturbed_action(self, observation, t):
        # Return residual action: U + noise (to be added to mu if available)
        residual = self.U[self.b, :, t] + self.noise[self.b, :, t]  # [K, nu]
        
        # Update noise to reflect any constraints applied later
        self.noise[self.b, :, t] = residual - self.U[self.b, :, t]

        return residual

    def _bound_action(self, action):
        """bound action within action space"""
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def _step_rollout(self, observation, action):
        """step the trajectory forward with the appropriate dynamics and cost models or environments"""
        if self.f is not None:
            return self.f(observation, action).to(self.dtype), self.l(observation, action).flatten().to(self.dtype)
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
        self.noise = torch.zeros(self.B, self.K, self.T+1, self.nu, dtype=self.dtype)
        self.value = torch.zeros(self.B, 1)
        self.observation = None
        self.rollout_observation = None
        self.rollout_observations = None
        self.rollout_actions = None

if __name__ == '__main__':
    import gymnasium as gym
    from networks import StageCost, Dynamics
    from env import BaseEnvWrapper, ClassicMPPIWrapper, MujocoMPPIWrapper

    B = 1
    P = 1
    K = 50

    env = BaseEnvWrapper(gym.make('InvertedPendulum-v5', render_mode='human'))
    ns = env.observation_space.shape[0]
    rollout_envs = gym.make_vec('InvertedPendulum-v5', num_envs=K, vectorization_mode="sync", wrappers=[MujocoMPPIWrapper])

    cov = 0.1*torch.diag(torch.ones(np.prod(env.action_space.shape)))
    mppi = MPPI(env, rollout_envs=rollout_envs, K=K, T=20, B=B, P=P, cov=cov, lambda_=0.1)

    obs, _ = env.reset()
    env.render()
    
    for _ in range(500):
        batch = np.repeat([obs[None,:]], B, axis=0)
        belief = batch + 0.*np.random.randn(B,P,ns)
        action = mppi.make_step(belief)
        value = mppi.get_value(belief)
            
        obs, reward, _, _, _ = env.step(action[0,0])
        print(reward)
        env.render()

    env.close()
    rollout_envs.close()