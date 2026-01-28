import os
import time
from dataclasses import dataclass

import gymnasium
import highway_env
from matplotlib import pyplot as plt

import random
import numpy as np
import torch

from env import BaseEnvWrapper, MPPIEnvWrapper
from mppi import MPPI
from networks import Dynamics, StageCost

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    env_id: str = "highway-fast-v0"
    """the environment id of the Atari game"""
    action_type: str = "ContinuousAction"
    """environment action space"""
    T: int = 5
    """length of MPPI rollout trajectories"""
    K: int = 2000
    """number of MPPI rollouts"""
    sync: bool = True # if using >100 particles on laptop, probably best False
    """whether to use Sync or AsyncVectorEnv"""

def make_env(env_id, action_type, seed, idx, capture_video, run_name):

    config = {
                # "observation": {
                #     "type": "Kinematics",
                #     "vehicles_count": 15,
                #     "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                #     "features_range": {
                #         "x": [-100, 100],
                #         "y": [-100, 100],
                #         "vx": [-20, 20],
                #         "vy": [-20, 20]
                #     },
                #     "absolute": True
                #     "order": "sorted"
                # },
                "action": {
                    "type": action_type,
                },
                "show_trajectories": True
            }

    def thunk():
        if capture_video and idx == 0:
            env = gymnasium.make(env_id, render_mode="rgb_array", config=config)
            env = BaseEnvWrapper(env)
            # env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gymnasium.make(env_id, config=config)
            env = MPPIEnvWrapper(env)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def plot_rollouts(states, actions, K, obs, action, next_obs):
    states = states[:,:-1].numpy()
    actions = actions.numpy()
    ts = np.arange(actions.shape[1])

    obs_states = np.array([obs.flatten(), next_obs.flatten()])

    fig, axs = plt.subplots(nrows=3, ncols=K, sharex=True, sharey=True)
    for k in range(K):
        axs[0,k].plot(ts, states[k,:,1:3])
        axs[1,k].plot(ts, states[k,:,3:5])
        axs[2,k].plot(ts, actions[k])

        axs[0,k].plot(ts[:2], obs_states[:,1:3], color='black')
        axs[1,k].plot(ts[:2], obs_states[:,3:5], color='black')
        axs[2,k].plot(ts[:1], [action], color='black', marker='*')

    axs[0,0].set_ylim(-1,1)

    axs[0,0].set_ylabel('pos')
    axs[1,0].set_ylabel('vel')
    axs[2,0].set_ylabel('a')

    plt.show()
    plt.pause(1)

def main(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    thunk = make_env(args.env_id, action_type=args.action_type, seed=args.seed, idx=0, capture_video=True, run_name=run_name)
    env = thunk()
    if args.sync:
        rollout_envs = gymnasium.vector.SyncVectorEnv([make_env(args.env_id, action_type=args.action_type, seed=args.seed, idx=i, capture_video=False, run_name=run_name) for i in range(args.K)])
    else:
        rollout_envs = gymnasium.vector.AsyncVectorEnv([make_env(args.env_id, action_type=args.action_type, seed=args.seed, idx=i, capture_video=False, run_name=run_name) for i in range(args.K)])

    # l = StageCost(env)
    # f = Dynamics(env)
    cov = 0.01*torch.diag(torch.ones(np.prod(env.action_space.shape), dtype=torch.float32))
    mppi = MPPI(env, l=None, f=None, rollout_envs=rollout_envs, K=args.K, T=args.T, cov=cov)

    obs, _ = env.reset(seed=args.seed)
    env.render()
    plt.pause(0.5)

    for _ in range(30):
        action = mppi.make_step(obs)
        next_obs, reward, done, truncated, info = env.step(action)

        # plot_rollouts(mppi.states, mppi.actions, mppi.K, obs, action, next_obs)

        obs = next_obs
        print(obs[0], action, reward)
        env.render()

    plt.imshow(env.render())

if __name__ == '__main__':
    args = Args()
    main(args)