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
    args.lambda_ = 0.1
    args.transition_network = "medium"
    args.transition_ensemble_size = 2
    args.mppi_handle_terminations = True
    args.trainer_scaler = 'standard'
    args.distributional_dynamics = False

    args.mppi_prior = 'gaussian'
    args.batch_size = 32
    args.horizon = 4
    args.num_rollouts = 300 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.1 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    # args.double_Q = True
    args.learning_rate = 1e-3
    args.episodic_learning = True
    for seed in range(5):
        for double_Q in [True, False]:
            args.double_Q = double_Q
            args.seed = seed
            train(args)