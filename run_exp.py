from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_project_name = "dual_mpcritic_ddpg_value_aligned_model_loss_v2"
    # args.env_id = "InvertedPendulum-v5"
    args.track = True
    args.save_model = True
    # args.total_timesteps = 100000
    # args.var = 1.0
    args.mppi_targets = False
    args.env_in_mppi = False
    args.num_rollouts = 200
    args.value_aligned_model_loss = True
    args.batch_size = 256
    args.transition_network = "InvertedPendulum"

for env in ['InvertedPendulum-v5']:
    if env == 'InvertedPendulum-v5':
        args.total_timesteps = 50000
    elif env == 'Hopper-v5':
        args.total_timesteps = 500000
    args.env_id = env
    for seed in range(5):
        args.seed = seed
        for mppi in [True]:
            args.mppi = mppi
            if mppi:
                for horizon in [4,1]:
                    args.horizon = horizon
                    for temp_model_loss in ["vaml"]:
                        args.temp_model_loss = temp_model_loss
                        for value_aligned_model_loss in [True]:
                            args.value_aligned_model_loss = value_aligned_model_loss
                            for control_mode in ["mu", "default"]:
                                args.mppi_control_mode = control_mode
                                run_name = train(args)
            else:
                args.horizon = 0
                run_name = train(args)
