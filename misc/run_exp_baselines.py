from collections.abc import Callable

from ddpg import Args as DDPGArgs
from ddpg import train as train_ddpg
from sac import Args as SACArgs
from sac import train as train_sac


ENV_TOTAL_TIMESTEPS = {
    "InvertedDoublePendulum-v5": 100_000,
    "Hopper-v5": 1_000_000,
    "Walker2d-v5": 1_000_000,
    "Ant-v5": 2_000_000,
}

SEEDS = range(10)

ALGORITHMS = {
    "ddpg": {
        "args_cls": DDPGArgs,
        "train_fn": train_ddpg,
        "wandb_project_name": "ddpg_baseline",
    },
    "sac": {
        "args_cls": SACArgs,
        "train_fn": train_sac,
        "wandb_project_name": "sac_baseline",
    },
}


def build_args(args_cls, env_id: str, total_timesteps: int, seed: int, wandb_project_name: str, wandb_entity: str):
    args = args_cls()
    args.env_id = env_id
    args.total_timesteps = total_timesteps
    args.seed = seed
    args.track = True
    args.save_model = True
    args.wandb_project_name = wandb_project_name
    args.wandb_entity = wandb_entity
    return args


if __name__ == "__main__":
    wandb_entity = ""
    for algorithm_name, algorithm_config in ALGORITHMS.items():
        args_cls = algorithm_config["args_cls"]
        train_fn: Callable = algorithm_config["train_fn"]
        wandb_project_name = algorithm_config["wandb_project_name"]

        for env_id, total_timesteps in ENV_TOTAL_TIMESTEPS.items():
            for seed in SEEDS:
                args = build_args(
                    args_cls=args_cls,
                    env_id=env_id,
                    total_timesteps=total_timesteps,
                    seed=seed,
                    wandb_project_name=wandb_project_name,
                    wandb_entity=wandb_entity
                )
                print(
                    f"Running {algorithm_name} on {env_id} with seed={seed} "
                    f"for {total_timesteps} timesteps"
                )
                train_fn(args)

