# from ddpg_continuous_action import Args, train
from ddpg_continuous_action_v2 import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_hopper_target_horizon_0"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Hopper-v5"

    # agreed upon
    args.total_timesteps = 500000
    args.lambda_ = 0.1
    args.transition_network = "medium"
    args.transition_ensemble_size = 2
    args.mppi_handle_terminations = False
    args.trainer_scaler = 'standard'
    args.distributional_dynamics = False

    args.mppi_prior = 'gaussian'
    args.batch_size = 64
    args.horizon = 4
    args.num_rollouts = 200 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.1 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    args.double_Q = False
    args.learning_rate = 1e-3
    # args.episodic_learning = True
    args.target_horizon = 0
    for seed in range(5):
        for horizon in [4,0]:
            args.horizon = horizon
            args.seed = seed
            train(args)