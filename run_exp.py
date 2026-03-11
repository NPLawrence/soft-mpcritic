from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_project_name = "dual_mpcritic_ddpg_redo_rewardmodel_full"
    # args.env_id = "InvertedPendulum-v5"
    args.track = True
    args.save_model = True
    # args.total_timesteps = 100000
    # args.var = 1.0
    args.mppi_targets = False
    args.env_in_mppi = False
    args.num_rollouts = 200
    # args.batch_size = 32
    # args.transition_network = "JointMLP_delta"

for env in ['InvertedPendulum-v5', 'Hopper-v5']:
    if env == 'InvertedPendulum-v5':
        args.total_timesteps = 50000
    elif env == 'Hopper-v5':
        args.total_timesteps = 500000
    args.env_id = env
    for batch_size in [256]:
        args.batch_size = batch_size
        for multiplier in [1.0, 10.0]:
            args.lambda_ = 0.1*multiplier
            args.var = 0.1*multiplier
            for seed in range(5):
                args.seed = seed
                for use_huber in [False,True]:
                    args.use_huber_loss = use_huber
                    for mppi in [True]:
                        args.mppi = mppi
                        if mppi:
                            for horizon in [1]:
                                args.horizon = horizon
                                for control_mode in ["mu", "default"]:
                                    args.mppi_control_mode = control_mode
                                    run_name = train(args)
                        else:
                            args.horizon = 0
                            run_name = train(args)
