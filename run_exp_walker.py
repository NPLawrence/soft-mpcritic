from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_terminations_walker"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Walker2d-v5"

    # agreed upon
    args.total_timesteps = 500000
    args.batch_size = 128
    args.horizon = 3
    args.num_rollouts = 500 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.05 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    args.lambda_ = 0.1
    args.transition_network = "medium"
    args.transition_ensemble_size = None
    args.mppi_handle_terminations = True
    args.trainer_scaler = 'standard'

    args.model_optimizer = 'soap'
    args.mppi_prior = 'gaussian'
    # args.mppi_prior = 'uniform'
    for seed in range(5):
        args.seed = seed
        for distributional_dynamics in [False]:
            args.distributional_dynamics = distributional_dynamics
            train(args)