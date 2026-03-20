from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value_with_discount"
    args.track = True
    args.save_model = True
    args.var = 0.1
    args.mppi = True
    args.mppi_targets = True
    args.env_in_mppi = False
    args.num_rollouts = 200
    args.horizon = 4
    args.target_horizon = 4
    args.num_target_rollouts = 20
    args.transition_network = "medium"
    
    # DIP - performance
    # args.env_id = "InvertedDoublePendulum-v5"
    # args.total_timesteps = 50000
    # for seed in range(10):
    #     args.seed = seed
    #     for transition_ensemble_size in [None, 1]:
    #         args.transition_ensemble_size = transition_ensemble_size
    #         for mppi_target_warmstart in [True, False]:    
    #             args.mppi_target_warmstart = mppi_target_warmstart
    #             if not args.mppi_target_warmstart:
    #                 for mppi_target_iterations in [1,5]:
    #                     args.mppi_target_iterations = mppi_target_iterations
    #                     train(args)
    #             else:
    #                 args.mppi_target_iterations = 5
    #                 if transition_ensemble_size is None and mppi_target_warmstart == True:
    #                     for target_horizon in [1,4]:
    #                         args.target_horizon = target_horizon
    #                         run_name = train(args)
    #                 else:
    #                     args.target_horizon = 4
    #                     run_name = train(args)

    # Hopper
    args.env_id = "Hopper-v5"
    args.total_timesteps = 500000
    args.transition_ensemble_size = None
    args.mppi_target_warmstart = True
    args.target_horizon = 4
    for seed in range(10):
        args.seed = seed
        for transition_ensemble_size in [None, 1]:
            args.transition_ensemble_size = transition_ensemble_size
            args.mppi_targets = True
            args.mppi_online = True
            train(args)
        for mppi_online in [False]: # [False, True]
            args.transition_ensemble_size = None
            args.mppi_targets = True
            args.mppi_online = mppi_online
            train(args)
        for mppi_targets in [False]:
            args.transition_ensemble_size = None
            args.mppi_online = True
            args.mppi_targets = mppi_targets
            train(args)

    # # DIP - speed
    # args.wandb_project_name = "dual_mpcritic_ddpg_sps"
    # args.env_id = "InvertedDoublePendulum-v5"
    # args.total_timesteps = 50000
    # args.transition_ensemble_size = None
    # args.target_horizon = 4
    # for seed in range(3):
    #     args.seed = seed
    #     for mppi_target_warmstart in [True, False]:    
    #         args.mppi_target_warmstart = mppi_target_warmstart
    #         for num_target_rollouts in [10, 20, 50]:
    #             args.num_target_rollouts = num_target_rollouts
    #             if not args.mppi_target_warmstart:
    #                 for mppi_target_iterations in [1,5]:
    #                     args.mppi_target_iterations = mppi_target_iterations
    #                     train(args)
    #             else:
    #                 args.mppi_target_iterations = 5
    #                 run_name = train(args)

    # args.env_id = "Ant-v5"
    # args.total_timesteps = 500000
    # args.transition_ensemble_size = None
    # args.mppi_target_warmstart = True

    # # args.target_horizon = 4
    # args.transition_network = "medium"
    # args.horizon = 8
    # args.target_horizon = 8
    # args.num_rollouts = 200
    # args.num_target_rollouts = 20
    # for seed in range(10):
    #     args.seed = seed
    #     train(args)