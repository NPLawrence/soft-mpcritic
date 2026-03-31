# soft-mpcritic

This is the codebase for our CDC paper `Soft MPCritic: Amortized model predictive value iteration`

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- MuJoCo-compatible environment (for Gymnasium MuJoCo tasks)

## Setup

```bash
uv venv
uv sync
```

Run commands through `uv run` so they use the project environment.

## Quick Start

Inspect available flags:

```bash
uv run python ddpg_continuous_action_v2.py --help
uv run python sac.py --help
```

Example single runs:

```bash
uv run python ddpg_continuous_action_v2.py \
	--env-id Hopper-v5 \
	--track False \
	--total-timesteps 500000 \
	--horizon 4 \
	--target-horizon 4 \
	--num-rollouts 200 \
	--num-target-rollouts 20

uv run python sac.py \
	--env-id Hopper-v5 \
	--track False \
	--total-timesteps 500000
```

## Experiment Scripts

The repository includes convenience launchers for sweeps and named experiment sets:

- `run_exp.py`
- `run_exp_walker.py`
- `run_exp_baselines.py`
- `run_exp_npl.py`
- `run_exp_soap.py`
- `run_exp_tb.py`

These scripts set arguments in code and execute repeated runs over seeds/hyperparameters.

## Outputs

- Run artifacts are saved under `runs/`.
- W&B logs are used when `track=True`.
- Plotting scripts are under `plotting/`.

## Repository Layout Notes

- Legacy or experimental code that is not part of the main paper workflows is placed under `misc/`.

## Reproducibility Notes

- The repository uses `uv.lock` for pinned dependencies.
- Most scripts expose a `seed` argument.
- MuJoCo/Gymnasium versions can affect benchmark numbers; keep versions consistent with `pyproject.toml`.

## Development Notes

- Install dependencies with `uv sync`.
- Run a quick import/training CLI check before sharing results.
- Keep large local artifacts (`runs/`, `wandb/`, generated figures) out of commits.

## License

This project is licensed under the terms of the MIT License. See `LICENSE`.
