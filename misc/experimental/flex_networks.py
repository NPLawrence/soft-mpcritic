import random
import numpy as np
import torch
import torch.nn as nn


activation_map = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


class JointMLP_flex(nn.Module):
    def __init__(self, env, num_hidden=2, num_nodes=256, activations=[nn.SiLU(), nn.SiLU()]):
        super().__init__()
        if len(activations) != num_hidden:
            raise ValueError(f"num_hidden must be == len(activations), got {num_hidden}.")

        self.nx = np.prod(env.observation_space.shape)
        self.nu = np.prod(env.action_space.shape)
        self.get_torch_reward = env.get_wrapper_attr('get_torch_reward')

        layers = [
            nn.Linear(self.nx + self.nu, num_nodes),
            np.random.choice(activations)
        ]
        for _ in range(num_hidden-1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(np.random.choice(activations))

        layers.append(nn.Linear(num_nodes, self.nx))

        self.net = nn.Sequential(*layers)

    def forward(self, x, u):
        z = torch.cat([x, u], 1)
        x_next = x + self.net(z)
        with torch.no_grad():
            reward = self.get_torch_reward(x, u, x_next)
        return x_next, reward


class FlexEnsembleDynamicsModel(nn.Module):
    def __init__(self, env, ensemble_size=5, num_hidden_list=None, num_nodes_list=None, activations_list=None):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}.")

        self.ensemble_size = ensemble_size
        if num_hidden_list is None:
            num_hidden_list = [np.random.choice([2,3]) for _ in range(ensemble_size)]
        if num_nodes_list is None:
            num_nodes_list = [np.random.choice([64,128,256]) for _ in range(ensemble_size)]
        if activations_list is None:
            activations_list = [
                [random.choice([nn.ReLU, nn.SiLU, nn.Tanh])() for _ in range(num_hidden_list[i])]
                for i in range(ensemble_size)
            ]
        elif activations_list[0][0] in activation_map.keys():
            activations_list = [
                [activation_map[activations_list[i][j]]() for j in range(num_hidden_list[i])]
                for i in range(ensemble_size)
            ]
        else:
            activations_list = [
                [activation_map[act]() for act in activations_list[i]]
                for i in range(ensemble_size)
            ]

        if len(num_hidden_list) != ensemble_size:
            raise ValueError(f"len(num_hidden_list) must equal ensemble_size={ensemble_size}, got {len(num_hidden_list)}.")
        if len(num_nodes_list) != ensemble_size:
            raise ValueError(f"len(num_nodes_list) must equal ensemble_size={ensemble_size}, got {len(num_nodes_list)}.")
        if len(activations_list) != ensemble_size:
            raise ValueError(f"len(activations_list) must equal ensemble_size={ensemble_size}, got {len(activations_list)}.")
        for i, acts in enumerate(activations_list):
            if len(acts) != num_hidden_list[i]:
                raise ValueError(
                    f"activations_list[{i}] has {len(acts)} entries but num_hidden_list[{i}]={num_hidden_list[i]}."
                )

        self.models = nn.ModuleList(
            [JointMLP_flex(env, num_hidden_list[i], num_nodes_list[i], activations_list[i]) for i in range(ensemble_size)]
        )

    def forward(self, x, u, model_indices=None):
        if model_indices is not None:
            if model_indices.shape[0] != x.shape[0]:
                raise ValueError(
                    f"model_indices must have shape ({x.shape[0]},), got {tuple(model_indices.shape)}."
                )
            model_indices = model_indices.to(device=x.device, dtype=torch.long)
            x_next = torch.empty_like(x)
            reward = None
            termination = None
            for model_idx in torch.unique(model_indices).tolist():
                mask = model_indices == model_idx
                member_x_next, member_reward, member_termination = self.models[model_idx](x[mask], u[mask])
                x_next[mask] = member_x_next
                if reward is None:
                    reward_shape = (x.shape[0],) + tuple(member_reward.shape[1:])
                    reward = torch.empty(reward_shape, device=member_reward.device, dtype=member_reward.dtype)
                if member_termination is not None and termination is None:
                    termination_shape = (x.shape[0],) + tuple(member_termination.shape[1:])
                    termination = torch.empty(termination_shape, device=member_termination.device, dtype=member_termination.dtype)
                reward[mask] = member_reward
                if termination is not None and member_termination is not None:
                    termination[mask] = member_termination
            return x_next, reward, termination

        model_idx = int(torch.randint(self.ensemble_size, (1,), device=x.device).item())
        return self.models[model_idx](x, u)
