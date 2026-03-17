from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_union_ensemble_unboundedQ"
    args.track = True
    args.save_model = True
    args.mppi = True
    args.env_in_mppi = False
    args.mppi_targets = True
    args.num_rollouts = 200
    args.batch_size = 64
    args.transition_network = "medium"
    args.use_huber_loss = False

    args.ensemble_rollout_mode = "trajectory"

    for env in ["Hopper-v5", "Walker2d-v5"]:
        if env == 'InvertedDoublePendulum-v5':
            args.total_timesteps = 50000
        elif env == 'Walker2d-v5' or env == 'Hopper-v5':
            args.total_timesteps = 500000
        elif env == 'Reacher-v5':
            args.total_timesteps = 50000
        args.env_id = env
        for horizon in [4]:
            args.horizon = horizon
            for seed in range(3):
                args.seed = seed
                for transition_ensemble_size in [None]:
                    args.transition_ensemble_size = transition_ensemble_size
                    

                    # args.mppi_control_mode = 'mean_residual'
                    # args.mppi_target_mode = 'mean_residual'
                    # # for horizon in [1,4]:
                    # # for horizon in [4,8]:
                    # #     args.horizon = horizon
                    # for num_target_rollouts in [10]:
                    #     args.num_target_rollouts = num_target_rollouts
                    #     train(args)

                    args.mppi_control_mode = 'default'
                    args.mppi_target_mode = 'default'
                    # for horizon in [1,4]:
                # for horizon in [4,8]:
                #         args.horizon = horizon
                    for num_target_rollouts in [10]:
                        args.num_target_rollouts = num_target_rollouts
                        train(args)

                #     args.mppi_control_mode = 'mean_residual'
                #     args.mppi_target_mode = 'mean_residual'
                #     # for horizon in [1,4]:
                # # for horizon in [4,8]:
                # #         args.horizon = horizon
                #     for num_target_rollouts in [10]:
                #         args.num_target_rollouts = num_target_rollouts
                #         train(args)

                #     args.mppi_control_mode = 'mu'
                #     args.mppi_target_mode = 'mu'
                #     # for horizon in [1,4]:
                # # for horizon in [4,8]:
                # #         args.horizon = horizon
                #     for num_target_rollouts in [10]:
                #         args.num_target_rollouts = num_target_rollouts
                #         train(args)
                # args.mppi_control_mode = 'mu'
                # args.mppi_target_mode = 'mu'
                # # for horizon in [1,4]:
                # for horizon in [1,4,8]:
                #     args.horizon = horizon
                #     for num_target_rollouts in [10,100]:
                #         args.num_target_rollouts = num_target_rollouts
                #         train(args)
