import gymnasium
from copy import deepcopy
from typing import TypeVar
import numpy as np

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