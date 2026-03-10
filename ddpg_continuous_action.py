# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

from gymppi.mppi import MPPI
from gymppi.env import BaseEnvWrapper, ClassicMPPIWrapper, MujocoMPPIWrapper
from gymppi.buffers import WarmstartReplayBuffer
from torch_utils.networks import JointMLP_sm, JointMLP_lg, JointMLP_InvPend
from torch_utils.trainer import Trainer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "dual_mpcritic"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "InvertedPendulum-v5"
    """the environment id"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 0 # 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""

    # MPPI arguments
    mppi: bool = True
    """use MPPI online (making extra envs)"""
    env_in_mppi: bool = True
    """use the environment for MPPI rollouts"""
    mu_in_mppi: bool = False
    """use policy mean in MPPI rollouts"""
    Q_in_mppi: bool = True
    """use Q-function as MPPI terminal cost"""
    mppi_targets: bool = True
    """use MPPI for Q-function targets"""
    mppi_target_warmstart: bool = True
    """warmstart MPPI for Q-function targets"""
    horizon: int = 1
    """length of MPPI rollouts/trajectories"""
    num_rollouts: int = 100
    """number of rollouts/trajectory samples for MPPI"""
    num_particles: int = 1
    """number of states/particles to rollout from"""
    var: float = 0.1
    """variance for noise in each action dimension"""
    lambda_: float = 0.1
    """temperature parameter in MPPI"""
    vectorization_mode: str = 'sync'
    """vectorization mode for gymnasium mppi rollouts"""
    transition_utd: int = 2
    """the frequency of training the transition model"""
    transition_network: str = 'small'
    """size/type of transition model network"""
    transition_batch_size: int = 32
    """batch size for transition model updates"""
    model_predict_delta: bool = True
    """train transition model on state deltas (next_obs - obs)"""
    use_huber_loss: bool = True
    """if True use Huber (SmoothL1) universally, else use MSE universally"""
    huber_delta: float = 1.0
    """SmoothL1 (Huber) transition point for both model/reward and Q losses"""


def make_env(env_id, seed, idx, capture_video, run_name, env_kwargs={}):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="human", **env_kwargs)
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = BaseEnvWrapper(env)
        env.action_space.seed(seed)
        return env

    return thunk

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.silu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def train(args):
    # args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if any(s in args.env_id for s in ['Swimmer', 'Hopper', 'Walker', 'Cheetah', 'Ant', 'Humanoid']):
        env_kwargs = {'exclude_current_positions_from_observation': False}
    else:
        env_kwargs = {}

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # see: https://farama.org/Vector-Autoreset-Mode and https://github.com/vwxyzjn/cleanrl/issues/499 for reference
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, env_kwargs)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)    
    if args.mppi and args.env_in_mppi:
        rollout_envs = gym.make_vec(args.env_id, num_envs=args.num_rollouts, vectorization_mode=args.vectorization_mode, wrappers=[MujocoMPPIWrapper], **env_kwargs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    model_loss = nn.SmoothL1Loss(beta=args.huber_delta) if args.use_huber_loss else nn.MSELoss()
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = Adam(list(actor.parameters()), lr=args.learning_rate)

    if args.mppi:
        Q = qf1 if args.Q_in_mppi else None
        mu = actor if args.mu_in_mppi else None
        cov = args.var*torch.diag(torch.ones(np.prod(envs.single_action_space.shape), dtype=torch.float32))

        if args.env_in_mppi:
            mppi = MPPI(env=envs.envs[0], rollout_envs=rollout_envs, gamma=args.gamma, Q=Q, mu=mu,
                        B=1, P=args.num_particles, T=args.horizon, K=args.num_rollouts,
                        lambda_=args.lambda_, cov=cov)
            if args.mppi_targets:
                target_mppi = MPPI(env=envs.envs[0], rollout_envs=rollout_envs, gamma=args.gamma, Q=qf1_target, mu=target_actor,
                                B=args.batch_size, P=args.num_particles, T=args.horizon, K=args.num_rollouts,
                                lambda_=args.lambda_, cov=cov)
        else:
            if args.transition_network == 'small':
                transition_model = JointMLP_sm(env=envs.envs[0])
            elif args.transition_network == 'large':
                transition_model = JointMLP_lg(env=envs.envs[0])
            elif args.transition_network == 'InvertedPendulum':
                transition_model = JointMLP_InvPend(env=envs.envs[0])
                
            transition_trainer = Trainer(
                transition_model,
                Adam,
                lr=args.learning_rate,
                model_loss=model_loss,
                predict_delta=args.model_predict_delta,
                huber_delta=args.huber_delta,
            )

            mppi = MPPI(env=envs.envs[0], transition_model=transition_model, gamma=args.gamma, Q=Q, mu=mu,
                        B=1, P=args.num_particles, T=args.horizon, K=args.num_rollouts,
                        lambda_=args.lambda_, cov=cov)
            if args.mppi_targets:
                target_mppi = MPPI(env=envs.envs[0], transition_model=transition_model, gamma=args.gamma, Q=qf1_target, mu=target_actor,
                                B=args.batch_size, P=args.num_particles, T=args.horizon, K=args.num_rollouts,
                                lambda_=args.lambda_, cov=cov)

    rb = WarmstartReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_U=args.horizon+1
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if args.capture_video:
        envs.envs[0].render()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            Us = np.empty([args.num_particles, args.horizon+1] + list(envs.single_action_space.shape))
        else:
            with torch.no_grad():
                if not args.mppi:
                    actions = actor(torch.Tensor(obs).to(dtype=torch.float32).to(device))
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
                    Us = np.empty([args.num_particles, args.horizon+1] + list(envs.single_action_space.shape))
                else:
                    # ensure obs is shape of B X P X 1 X S, assumes B=1
                    actions = mppi.make_step(obs.reshape(1, args.num_particles, 1, -1))[0]
                    Us = mppi.U[0].cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if args.capture_video:
            envs.envs[0].render()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            # for info in infos["final_info"]:
            #     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            for idx, final in enumerate(np.logical_or(terminations, truncations)):
                print(f"global_step={global_step}, episodic_return={infos['final_info']['episode']['r'][idx]}")
                writer.add_scalar("charts/episodic_return", infos['final_info']["episode"]["r"][idx], global_step)
                writer.add_scalar("charts/episodic_length", infos['final_info']["episode"]["l"][idx], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        for idx, final in enumerate(np.logical_or(terminations, truncations)):
            if final:
                real_next_obs[idx] = infos["final_obs"][idx]
                if args.mppi:
                    mppi.reset()
        rb.add(obs, real_next_obs, actions, Us, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step == args.learning_starts and args.mppi and not args.env_in_mppi:
            for _ in range(int(args.learning_starts*args.transition_utd)):
                data, _, _ = rb.sample(args.batch_size)
                dynamics_loss, reward_loss = transition_trainer.update(data)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data, batch_inds, env_indices = rb.sample(args.batch_size)
            with torch.no_grad():
                if args.mppi and args.mppi_targets:
                    mppi_next_observations = data.next_observations.reshape(args.batch_size, args.num_particles, 1, -1)
                    U_init = data.Us if args.mppi_target_warmstart else None

                    # target_mppi.reset(U_init)
                    # qf1_next_target, next_Us = target_mppi.get_value(mppi_next_observations)
                    qf1_next_target, next_Us = target_mppi.get_value(mppi_next_observations, U_init)

                    rb.Us[batch_inds, env_indices, 1:] = next_Us[:,0,:-1] # copy over solution offset by 1 step
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
                else:
                    next_state_actions = target_actor(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            if args.use_huber_loss:
                qf1_loss = F.smooth_l1_loss(qf1_a_values, next_q_value, beta=args.huber_delta)
            else:
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # if not args.env_in_mppi and (global_step % args.transition_frequency == 0):
            if args.mppi and not args.env_in_mppi:
                dynamics_loss, reward_loss = transition_trainer.update(data)
                for _ in range(args.transition_utd-1):
                    data, _, _ = rb.sample(args.batch_size)
                    dynamics_loss, reward_loss = transition_trainer.update(data)
                # for _ in range(args.transition_utd):
                #     data, _, _ = rb.sample(args.transition_batch_size)
                #     dynamics_loss, reward_loss = transition_trainer.update(data)
            else:
                reward_loss = torch.tensor([0.])
                dynamics_loss = torch.tensor([0.])

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/dynamics_loss", dynamics_loss.item(), global_step)
                writer.add_scalar("losses/reward_loss", reward_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        # from cleanrl_utils.evals.ddpg_eval import evaluate

        # episodic_returns = evaluate(
        #     model_path,
        #     make_env,
        #     args.env_id,
        #     eval_episodes=10,
        #     run_name=f"{run_name}-eval",
        #     Model=(Actor, QNetwork),
        #     device=device,
        #     exploration_noise=args.exploration_noise,
        # )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    if args.track:
        wandb.finish()
    envs.close()
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
