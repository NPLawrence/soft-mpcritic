from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value_with_discount_iterations_walker"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Walker2d-v5"

    # agreed upon
    args.total_timesteps = 500000
    args.batch_size = 64
    args.horizon = 3
    args.num_rollouts = 200 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.1 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    args.lambda_ = 0.1

    args.transition_network = "small_deep"
    args.transition_ensemble_size = None

    for seed in range(5):
        args.seed = seed
        for mppi_online_iterations in [1,2,4]:
            args.mppi_online_iterations = mppi_online_iterations
            train(args)