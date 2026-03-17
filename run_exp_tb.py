from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value"
    args.track = True
    args.save_model = True
    args.var = 0.1
    args.mppi = True
    args.mppi_targets = True
    args.num_target_rollouts = 10
    args.env_in_mppi = False
    args.num_rollouts = 100
    args.horizon = 4
    args.target_horizon = 4
    
    args.env_id = "InvertedDoublePendulum-v5"
    args.total_timesteps = 50000
    for seed in range(5):
        args.seed = seed
        for transition_network in ["medium"]:
            args.transition_network = transition_network
            for transition_ensemble_size in [None, 1]:
                args.transition_ensemble_size = transition_ensemble_size
                for mppi_target_warmstart in [True, False]:    
                    args.mppi_target_warmstart = mppi_target_warmstart
                    if not args.mppi_target_warmstart:
                        for mppi_target_iterations in [1,5,10]:
                            args.mppi_target_iterations = mppi_target_iterations
                            train(args)
                    else:
                        args.mppi_target_iterations = 5
                        run_name = train(args)

    # args.env_id = "Hopper-v5"
    # args.total_timesteps = 500000
    # args.transition_network = "medium"
    # args.transition_ensemble_size = None
    # args.mppi_target_warmstart = True
    # args.mppi_target_iterations = 5 # unused
    # for seed in range(3):
    #     args.seed = seed
    #     for horizon in [1,4,8]:
    #         args.horizon = horizon
    #         train(args)

    # args.env_id = "Walker2d-v5"
    # for seed in range(3):
    #     args.seed = seed
    #     for horizon in [1,4,8]:
    #         args.horizon = horizon
    #         train(args)