from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_union_refactor_mppi"
    args.track = True
    args.save_model = True
    args.mppi = True
    args.env_in_mppi = False
    args.mppi_targets = True
    args.num_rollouts = 100
    args.batch_size = 32
    args.transition_network = "medium"

    for env in ['InvertedPendulum-v5', 'HalfCheetah-v5']:
        if env == 'InvertedPendulum-v5':
            args.total_timesteps = 50000
        elif env == 'HalfCheetah-v5':
            args.total_timesteps = 500000
        args.env_id = env
        for horizon in [8,4]:
            args.horizon = horizon
            for seed in range(5):
                args.seed = seed

                args.mppi_control_mode = 'default'
                args.mppi_target_mode = 'default'
                # for horizon in [1,4]:
                # for horizon in [4,8]:
                #     args.horizon = horizon
                for num_target_rollouts in [10,100]:
                    args.num_target_rollouts = num_target_rollouts
                    train(args)

                args.mppi_control_mode = 'warmstart_residual'
                args.mppi_target_mode = 'warmstart_residual'
                # for horizon in [1,4]:
            # for horizon in [4,8]:
            #         args.horizon = horizon
                for num_target_rollouts in [10,100]:
                    args.num_target_rollouts = num_target_rollouts
                    train(args)

                args.mppi_control_mode = 'mean_residual'
                args.mppi_target_mode = 'mean_residual'
                # for horizon in [1,4]:
            # for horizon in [4,8]:
            #         args.horizon = horizon
                for num_target_rollouts in [10,100]:
                    args.num_target_rollouts = num_target_rollouts
                    train(args)

                args.mppi_control_mode = 'mu'
                args.mppi_target_mode = 'mu'
                # for horizon in [1,4]:
            # for horizon in [4,8]:
            #         args.horizon = horizon
                for num_target_rollouts in [10,100]:
                    args.num_target_rollouts = num_target_rollouts
                    train(args)
            # args.mppi_control_mode = 'mu'
            # args.mppi_target_mode = 'mu'
            # # for horizon in [1,4]:
            # for horizon in [1,4,8]:
            #     args.horizon = horizon
            #     for num_target_rollouts in [10,100]:
            #         args.num_target_rollouts = num_target_rollouts
            #         train(args)
