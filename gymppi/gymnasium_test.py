import os
import time
from dataclasses import dataclass

import gymnasium
from matplotlib import pyplot as plt

import random
import numpy as np
import torch

from env import MPPIEnvWrapper, BaseEnvWrapper
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
    
    env_id: str = 'InvertedPendulum-v5' # 'Reacher-v5' # 'InvertedDoublePendulum-v5' # 'InvertedPendulum-v5' # "MountainCarContinuous-v0" # "Pendulum-v1"
    # reward_control_weight in reacher is a bit of a nuisance, can set to 0 when calling gymnasium.make for pure tracking objective
    """the environment id of the Atari game"""
    action_type: str = "ContinuousAction"
    """environment action space"""
    T: int = 20 # 50 # 20 # 20 # 200 # 20
    """length of MPPI rollout trajectories"""
    K: int = 100 # 2000 # 2000 # 100 # 100 # 20
    """number of MPPI rollout trajectories"""
    sync: bool = True # if using >100 particles on laptop, probably best True
    """whether to use Sync or AsyncVectorEnv"""

def make_env(env_id, action_type, seed, idx, capture_video, run_name):

    def thunk():
        if capture_video and idx == 0:
            env = gymnasium.make(env_id, render_mode="human") #, reward_control_weight=0)
            env = BaseEnvWrapper(env)
            # env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gymnasium.make(env_id) # , reward_control_weight=0)
            env = MPPIEnvWrapper(env)
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

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
    cov = None if ('Pendulum-v5' not in args.env_id) else 0.01*torch.diag(torch.ones(np.prod(env.action_space.shape), dtype=torch.float64))
    mppi = MPPI(env, l=None, f=None, rollout_envs=rollout_envs, K=args.K, T=args.T, cov=cov)

    obs, _ = env.reset(seed=args.seed)
    env.render()
    plt.pause(0.5)

    for _ in range(500):
        action = mppi.make_step(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            env.reset()
            mppi.reset()
        print(action, reward)
        env.render()

if __name__ == '__main__':
    args = Args()
    main(args)