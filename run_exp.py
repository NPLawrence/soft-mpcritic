from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_project_name = "dual_mpcritic_ddpg_hopper"
    args.env_id = "Hopper-v5"
    args.track = True
    args.save_model = True
    args.total_timesteps = 500000
    # args.var = 1.0
    args.mppi_targets = False
    args.env_in_mppi = False
    args.num_rollouts = 200
    # args.batch_size = 32

for batch_size in [256]:
    args.batch_size = batch_size
    for seed in range(3):
        args.seed = seed
        for multiplier in [10.0]:
            args.lambda_ = 0.1*multiplier
            args.var = 0.1*multiplier
            for mppi in [True]:
                args.mppi = mppi
                if mppi:
                    for horizon in [1]:
                        args.horizon = horizon
                        for mu_in_mppi in [True, False]:
                            args.mu_in_mppi = mu_in_mppi
                            run_name = train(args)
                else:
                    run_name = train(args)
