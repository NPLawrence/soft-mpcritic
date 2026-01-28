import gymnasium
from copy import deepcopy
from typing import TypeVar
import numpy as np

# reference:
# https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/envs/common/abstract.py
# BaseEnvWrapper used for actual environment the agent interacts with
# - has state attribute for reliable MPPI
# MMPIEnvWrapper used for simulating MPPI rollouts
# - has unique step to prevent rollout termination/truncation state initialization
# - has unique reset to set initial state for reliable MPPI

Observation = TypeVar("Observation")

class BaseEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)

    @property
    def _state(self):
        if 'classic' in self.env.unwrapped.spec.entry_point:
            return self.env.unwrapped.state
        elif 'mujoco' in self.env.unwrapped.spec.entry_point:
            state = self.env.unwrapped.state_vector()
            ... # other environments beyond DIP may need more items to fully describe state
            if 'reacher' in self.env.unwrapped.spec.entry_point:
                state = np.concat([state, self.env.unwrapped.goal])
            return state
        elif 'highway' in self.env.unwrapped.spec.entry_point:
            return self.env.unwrapped.road

class MPPIEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated, truncated = False, False
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:

        obs, info = super().reset(seed=seed, options=options)

        if (options is not None) and ('classic' in self.env.unwrapped.spec.entry_point):
            self.env.unwrapped.state = deepcopy(options['state'])
            try: # Pendulum-v1
                obs = self.env.unwrapped._get_obs()
            except: # MountainCar
                obs = self.env.unwrapped.state
            info = {}
        elif (options is not None) and ('mujoco' in self.env.unwrapped.spec.entry_point):
            n_qpos = np.prod(self.env.unwrapped.data.qpos.shape)
            n_qvel = np.prod(self.env.unwrapped.data.qvel.shape)
            self.env.unwrapped.data.qpos = deepcopy(options['state'][:n_qpos])
            self.env.unwrapped.data.qvel = deepcopy(options['state'][n_qpos:n_qpos+n_qvel])
            ... # other environments may need more items
            if 'reacher' in self.env.unwrapped.spec.entry_point:
                self.env.unwrapped.goal = options['state'][n_qpos+n_qvel:]
            obs = self.env.unwrapped._get_obs()
            info = self.env.unwrapped._get_reset_info()
        if (options is not None) and ("highway" in self.env.unwrapped.spec.entry_point):
            # under development...
            self.env.unwrapped.define_spaces()  # First, to set the controlled vehicle class depending on action space
            self.env.unwrapped.time = self.env.unwrapped.steps = 0
            self.env.unwrapped.done = False
            road = deepcopy(options['state'])
            # vs = road.vehicles
            # keys = road.vehicles[0].__dict__.keys()
            # for i, v in enumerate(self.env.unwrapped.road.vehicles):
            #     for k in keys:
            #         setattr(v, k, getattr(vs[i], k))
            self.env.unwrapped.road = road
            self.env.unwrapped.define_spaces()
            obs = self.env.unwrapped.observation_type.observe()
            # print(obs)
            info = self.env.unwrapped._info(obs, action=self.env.unwrapped.action_space.sample())
            if self.env.unwrapped.render_mode == "human":
                self.render()

        return obs, info