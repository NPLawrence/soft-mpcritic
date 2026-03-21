import numpy as np
import pandas as pd
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
project = 'dual_mpcritic_ddpg_new_value_with_discount'
runs = api.runs(f"mpcritic-dpc/{project}")

# summary_list, config_list, name_list = [], [], []
run_name_list = []
env_id_list = []
mppi_online_list = []
Q_in_mppi_list = []
num_rollouts_list = []
horizon_list = []
mppi_targets_list = []
num_target_rollouts_list = []
target_horizon_list = []
transition_ensemble_size_list = []
mppi_target_warmstart_list = []
mppi_target_iterations_list = []
episodic_return_list = []
global_step_list = []
sps_list = []
dataset = {
    "run_name": run_name_list,
    "env_id": env_id_list,
    "mppi_online": mppi_online_list,
    "Q_in_mppi": Q_in_mppi_list,
    "num_rollouts": num_rollouts_list,
    "horizon": horizon_list,
    "mppi_targets": mppi_targets_list,
    "num_target_rollouts": num_target_rollouts_list,
    "target_horizon": target_horizon_list,
    "transition_ensemble_size": transition_ensemble_size_list,
    "mppi_target_warmstart": mppi_target_warmstart_list,
    "mppi_target_iterations": mppi_target_iterations_list,
    "episodic_return": episodic_return_list,
    "global_step": global_step_list,
    "sps": sps_list
}
for (i, run) in enumerate(runs):
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    # summary_list.append(run.summary._json_dict)

    if run.state == 'finished':

        # .config contains the hyperparameters.
        config = {k: v for k,v in run.config.items()
            if not k.startswith('_')}

        run_name_list.append(run.name)
        env_id_list.append(config['env_id'])
        mppi_online_list.append(config['mppi_online'])
        Q_in_mppi_list.append(config['Q_in_mppi'])
        num_rollouts_list.append(config['num_rollouts'])
        horizon_list.append(config['horizon'])
        mppi_targets_list.append(config['mppi_targets'])
        num_target_rollouts_list.append(config['num_target_rollouts'])
        target_horizon_list.append(config['target_horizon'])
        transition_ensemble_size_list.append(config['transition_ensemble_size'])
        mppi_target_warmstart_list.append(config['mppi_target_warmstart'])
        mppi_target_iterations_list.append(config['mppi_target_iterations'])

        history = run.scan_history(keys=["charts/episodic_return","global_step"])
        episodic_return = [row["charts/episodic_return"] for row in history if row["charts/episodic_return"] is not None]
        episodic_return_list.append(np.array(episodic_return))
        global_step = [row["global_step"] for row in history if row["global_step"] is not None]
        global_step_list.append(np.array(global_step))

        history = run.scan_history(keys=["charts/SPS"])
        sps = [row["charts/SPS"] for row in history if row["charts/SPS"] is not None]
        sps_list.append(np.array(sps))
        print(i)

runs_df = pd.DataFrame(dataset)
runs_df.to_pickle(f"data/{project}_all_data.pkl")