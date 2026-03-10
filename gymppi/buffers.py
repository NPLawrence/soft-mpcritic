import warnings
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

class WarmstartReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    Us: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # For n-step replay buffer
    discounts: th.Tensor | None = None

class WarmstartReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    Us: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        n_U: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.n_U = n_U

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # # Check that the replay buffer can fit into the memory
        # if psutil is not None:
        #     mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.Us = np.zeros(
            (self.buffer_size, self.n_envs, self.n_U, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # if psutil is not None:
        #     total_memory_usage: float = (
        #         self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        #     )

        #     if not optimize_memory_usage:
        #         total_memory_usage += self.next_observations.nbytes

        #     if total_memory_usage > mem_available:
        #         # Convert to GB
        #         total_memory_usage /= 1e9
        #         mem_available /= 1e9
        #         warnings.warn(
        #             "This system does not have apparently enough memory to store the complete "
        #             f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
        #         )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        U: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Any,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        U = U.reshape((self.n_envs, self.n_U, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.Us[self.pos] = np.array(U)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> WarmstartReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> WarmstartReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env).astype(np.float32),
            self.actions[batch_inds, env_indices, :].astype(np.float32),
            self.Us[batch_inds, env_indices, :, :].astype(np.float32),
            next_obs.astype(np.float32),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1).astype(np.float32),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env).astype(np.float32),
        )
        return WarmstartReplayBufferSamples(*tuple(map(self.to_torch, data))), batch_inds, env_indices

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike | None) -> np.typing.DTypeLike | None:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype

# class WarmstartReplayBuffer(object):
#     def __init__(self, size: int):
#         """
#         Implements a ring buffer (FIFO).

#         :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
#             memories are dropped.
#         """
#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0

#     def __len__(self) -> int:
#         return len(self._storage)

#     @property
#     def storage(self):
#         """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
#         return self._storage

#     @property
#     def buffer_size(self) -> int:
#         """float: Max capacity of the buffer"""
#         return self._maxsize

#     def can_sample(self, n_samples: int) -> bool:
#         """
#         Check if n_samples samples can be sampled
#         from the buffer.

#         :param n_samples: (int)
#         :return: (bool)
#         """
#         return len(self) >= n_samples

#     def is_full(self) -> int:
#         """
#         Check whether the replay buffer is full or not.

#         :return: (bool)
#         """
#         return len(self) == self.buffer_size

#     def add(self, obs_t, action, reward, obs_tp1, done, U=None):
#         """
#         add a new transition to the buffer

#         :param obs_t: (Union[np.ndarray, int]) the last observation
#         :param action: (Union[np.ndarray, int]) the action
#         :param reward: (float) the reward of the transition
#         :param obs_tp1: (Union[np.ndarray, int]) the current observation
#         :param done: (bool) is the episode done
#         """
#         data = (obs_t, action, reward, obs_tp1, done, U)

#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize

#     def extend(self, obs_t, action, reward, obs_tp1, done, U):
#         """
#         add a new batch of transitions to the buffer

#         :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
#         :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
#         :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
#         :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
#         :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch
#         :param U: (Union[Tuple[float], np.ndarray]) MPPI action sequence

#         Note: uses the same names as .add to keep compatibility with named argument passing
#                 but expects iterables and arrays with more than 1 dimensions
#         """
#         for data in zip(obs_t, action, reward, obs_tp1, done, U):
#             if self._next_idx >= len(self._storage):
#                 self._storage.append(data)
#             else:
#                 self._storage[self._next_idx] = data
#             self._next_idx = (self._next_idx + 1) % self._maxsize

#     @staticmethod
#     def _normalize_obs(obs: np.ndarray,
#                        env: Optional[VecNormalize] = None) -> np.ndarray:
#         """
#         Helper for normalizing the observation.
#         """
#         if env is not None:
#             return env.normalize_obs(obs)
#         return obs

#     @staticmethod
#     def _normalize_reward(reward: np.ndarray,
#                           env: Optional[VecNormalize] = None) -> np.ndarray:
#         """
#         Helper for normalizing the reward.
#         """
#         if env is not None:
#             return env.normalize_reward(reward)
#         return reward

#     def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
#         obses_t, actions, rewards, obses_tp1, dones, Us = [], [], [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done, U = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#             dones.append(done)
#             Us.append(np.array(U, copy=False))
#         return (self._normalize_obs(np.array(obses_t), env),
#                 np.array(actions),
#                 self._normalize_reward(np.array(rewards), env),
#                 self._normalize_obs(np.array(obses_tp1), env),
#                 np.array(dones),
#                 np.array(Us))

#     def sample(self, batch_size: int, env: Optional[VecNormalize] = None, **_kwargs):
#         """
#         Sample a batch of experiences.

#         :param batch_size: (int) How many transitions to sample.
#         :param env: (Optional[VecNormalize]) associated gym VecEnv
#             to normalize the observations/rewards when sampling
#         :return:
#             - obs_batch: (np.ndarray) batch of observations
#             - act_batch: (numpy float) batch of actions executed given obs_batch
#             - rew_batch: (numpy float) rewards received as results of executing act_batch
#             - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
#             - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
#                 and 0 otherwise.
#         """
#         idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes, env=env)