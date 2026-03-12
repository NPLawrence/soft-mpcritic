import gymnasium
from copy import deepcopy
from typing import TypeVar
import numpy as np
import torch

Observation = TypeVar("Observation")

class BaseEnvWrapper(gymnasium.Wrapper):
    """base environment that the agent interacts with online"""
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    @property
    def _extra(self):
        """extra stuff not in observations necessary to represent the full system state"""
        if 'mujoco' in self.env.unwrapped.spec.entry_point:
            if ':Swimmer' in self.env.unwrapped.spec.entry_point:
                return {'x_position':self.env.unwrapped.data.qpos[[0]],
                        'y_position':self.env.unwrapped.data.qpos[[1]]}
            if ':Reacher' in self.env.unwrapped.spec.entry_point:
                return {'goal_position':self.env.unwrapped.data.qpos[-2:],
                        'goal_velocity':self.env.unwrapped.data.qvel[-2:]}
            if ':Hopper' in self.env.unwrapped.spec.entry_point:
                # current velocity is clipped when getting observation
                return {'current_position':self.env.unwrapped.data.qpos[[0]],
                        'current_velocity':self.env.unwrapped.data.qvel}
            if ':Cheetah' in self.env.unwrapped.spec.entry_point:
                return {'current_position':self.env.unwrapped.data.qpos[[0]]}
            
    @property
    def reward_bounds(self):
        """approximate bounds on the reward for all environments"""
        if 'mujoco' in self.env.unwrapped.spec.entry_point:
            if ':InvertedPendulum' in self.env.unwrapped.spec.entry_point:
                return {'low': 0, 'high': 1}
            if ':InvertedDoublePendulum' in self.env.unwrapped.spec.entry_point:
                return {'low': -20, 'high': 10} # rough lower bound estimate (upper exact)
            if ':Swimmer' in self.env.unwrapped.spec.entry_point:
                return {'low': -5, 'high': 5} # very rough on both bound estimates
            if ':Reacher' in self.env.unwrapped.spec.entry_point:
                return {'low': -3, 'high': 0} # rough lower bound estimate (upper exact)
            if ':Hopper' in self.env.unwrapped.spec.entry_point:
                return {'low': -10, 'high': 10} # very rough on both bound estimates
            
    def get_torch_reward(self, obs, action, next_obs):
        # undecided if needing with torch.no_grad(): 
        if 'mujoco' in self.env.unwrapped.spec.entry_point:
            if ':InvertedPendulum' in self.env.unwrapped.spec.entry_point:
                return (torch.abs(next_obs[...,[1]]) <= 0.2).float()
            if ':InvertedDoublePendulum' in self.env.unwrapped.spec.entry_point:
                # x, _, y = self.data.site_xpos[0]
                x_cart = next_obs[...,[0]]
                sin_theta1 = next_obs[...,[1]]
                sin_theta2 = next_obs[...,[2]]
                cos_theta1 = next_obs[...,[3]]
                cos_theta2 = next_obs[...,[4]]
                theta1 = torch.arctan2(sin_theta1, cos_theta1)
                theta2 = torch.arctan2(sin_theta2, cos_theta2)
                x = x_cart + 0.6*sin_theta1 + 0.6*(torch.sin(theta1+theta2))
                y = 0.6*cos_theta1 + 0.6*(torch.cos(theta1+theta2))
                # v1, v2 = self.data.qvel[1:3]
                v1 = next_obs[...,[6]] # these are slighlty wrong because observation clips velocities to [-10,10]
                v2 = next_obs[...,[7]] # these are slighlty wrong because observation clips velocities to [-10,10]
                dist_penalty = 0.01 * x**2 + (y - 2) ** 2
                vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
                alive_bonus = self.env.unwrapped._healthy_reward * (y > 1)
                return alive_bonus - dist_penalty - vel_penalty
            if ':Swimmer' in self.env.unwrapped.spec.entry_point:
                # assumes self.env.unwrapped._exclude_current_positions_from_observation = False
                x_position_before = obs[...,[0]]
                x_position_after = next_obs[...,[0]]
                x_velocity = (x_position_after - x_position_before) / self.env.unwrapped.dt
                forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
                ctrl_cost = self.env.unwrapped._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1, keepdim=True)
                return forward_reward - ctrl_cost
            if ':Reacher' in self.env.unwrapped.spec.entry_point:
                # assumes self.env.unwrapped._exclude_current_positions_from_observation = False                ...
                vec = next_obs[...,-2:]
                reward_dist = -torch.linalg.norm(vec, dim=-1, keepdim=True) * self.env.unwrapped._reward_dist_weight
                reward_ctrl = -torch.sum(torch.square(action), dim=-1, keepdim=True) * self.env.unwrapped._reward_control_weight
                return reward_dist + reward_ctrl
            if ':Hopper' in self.env.unwrapped.spec.entry_point:
                # assumes self.env.unwrapped._exclude_current_positions_from_observation = False
                x_position_before = obs[...,[0]]
                x_position_after = next_obs[...,[0]]
                x_velocity = (x_position_after - x_position_before) / self.env.unwrapped.dt

                z = next_obs[...,[1]]
                angle = next_obs[...,[2]]
                state = next_obs[...,2:]

                min_state, max_state = self.env.unwrapped._healthy_state_range
                min_z, max_z = self.env.unwrapped._healthy_z_range
                min_angle, max_angle = self.env.unwrapped._healthy_angle_range

                healthy_state = torch.all(torch.logical_and(min_state < state, state < max_state), dim=-1, keepdim=True)
                healthy_z = torch.logical_and(min_z < z, z < max_z)
                healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
                is_healthy = torch.all(torch.concat([healthy_state, healthy_z, healthy_angle], dim=-1), dim=-1, keepdim=True)

                forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
                healthy_reward = self.env.unwrapped._healthy_reward * is_healthy
                ctrl_cost = self.env.unwrapped._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1, keepdim=True)

                return forward_reward + healthy_reward - ctrl_cost
            if ':Cheetah' in self.env.unwrapped.spec.entry_point:
                x_position_before = obs[...,[0]]
                x_position_after = next_obs[...,[0]]
                x_velocity = (x_position_after - x_position_before) / self.env.unwrapped.dt
                forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
                ctrl_cost = self.env.unwrapped._ctrl_cost_weight * torch.sum(torch.square(action), dim=-1, keepdim=True)

                return forward_reward - ctrl_cost


class ClassicMPPIWrapper(gymnasium.Wrapper):
    """environment used for MPPI rollouts for classic control environments"""
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.env_id = env.unwrapped.spec.id

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated, truncated = False, False # don't stop MPPI rollouts early
        return obs.astype(np.float32), reward, terminated, truncated, info
    
    def obs2state(self, observation):
        if self.env_id == 'Pendulum-v1':
            cos_theta = observation[...,[0]]
            sin_theta = observation[...,[1]]
            theta_dot = observation[...,[2]]
            theta = np.arctan2(sin_theta, cos_theta)
            state = np.concatenate([theta, theta_dot], axis=-1)
        elif self.env_id == 'MountainCarContinuous-v0':
            state = observation

        return state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:

        obs, info = super().reset(seed=seed, options=options)

        if options.get('observation') is not None:
            state = self.obs2state(options.get('observation'))
        self.env.unwrapped.state = deepcopy(state)

        if self.env_id == 'Pendulum-v1':
            observation = self.env.unwrapped._get_obs()
        elif self.env_id == 'MountainCarContinuous-v0':
            observation = self.env.unwrapped.state

        info = {}

        return observation.astype(np.float32), info


class MujocoMPPIWrapper(gymnasium.Wrapper):
    """environment used for MPPI rollouts for Mujoco environments"""
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.env_id = self.env.unwrapped.spec.id
        self.n_qpos = np.prod(self.env.unwrapped.data.qpos.shape)
        self.n_qvel = np.prod(self.env.unwrapped.data.qvel.shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated, truncated = False, False # don't stop MPPI rollouts early
        return obs.astype(np.float32), reward, terminated, truncated, info

    def obs2state(self, observation, extra=None):
        if self.env_id == 'InvertedPendulum-v5':
            qpos = observation[:self.n_qpos]
            qvel = observation[self.n_qpos:self.n_qpos+self.n_qvel]

        if self.env_id == 'InvertedDoublePendulum-v5':
            cart_pos = observation[[0]]
            sin_theta1 = observation[[1]]
            sin_theta2 = observation[[2]]
            cos_theta1 = observation[[3]]
            cos_theta2 = observation[[4]]
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            qpos = np.concatenate([cart_pos, theta1, theta2], axis=-1)
            qvel = observation[5:5+self.n_qvel]

        if self.env_id == 'Swimmer-v5':
            if self.env.unwrapped._exclude_current_positions_from_observation:
                qpos = np.concatenate([extra['x_position'], extra['y_position'], observation[:3]])
                qvel = observation[3:3+self.n_qvel]
            else:
                qpos = observation[:self.n_qpos]
                qvel = observation[self.n_qpos:self.n_qpos+self.n_qvel]

        if self.env_id == 'Reacher-v5':
            cos_theta1 = observation[[0]]
            cos_theta2 = observation[[1]]
            sin_theta1 = observation[[2]]
            sin_theta2 = observation[[3]]
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            qpos = np.concatenate([theta1, theta2, extra['goal_position']])
            qvel = np.concatenate([observation[6:8], extra['goal_velocity']])

        if self.env_id == 'Hopper-v5':
            if self.env.unwrapped._exclude_current_positions_from_observation:
                qpos = np.concatenate([extra['current_position'], observation[:5]])
                # current velocity is clipped when getting observation
                qvel = extra['current_velocity']
            else:
                qpos = observation[:self.n_qpos]
                # current velocity is clipped when getting observation
                qvel = extra['current_velocity']

        if self.env_id == 'HalfCheetah-v5':
            if self.env.unwrapped._exclude_current_positions_from_observation:
                qpos = np.concatenate([extra['current_position'], observation[:self.n_qpos-1]])
                qvel = observation[self.n_qpos-1:self.n_qpos-1+self.n_qvel]
            else:
                qpos = observation[:self.n_qpos]
                qvel = observation[self.n_qpos:self.n_qpos+self.n_qvel]

        return qpos, qvel

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:

        obs, info = super().reset(seed=seed, options=options)

        if options.get('extra') is not None:
            extra = deepcopy(options.get('extra'))
        else:
            extra = None

        if options.get('observation') is not None:
            qpos, qvel = self.obs2state(options.get('observation'), extra)

        self.env.unwrapped.set_state(deepcopy(qpos), deepcopy(qvel))

        if self.env_id == 'reacher':
            self.env.unwrapped.goal = deepcopy(extra['goal_position'])
        ... # other environments may need more items

        obs = self.env.unwrapped._get_obs()
        info = self.env.unwrapped._get_reset_info()

        return obs.astype(np.float32), info