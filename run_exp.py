from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_project_name = "dual_mpcritic_ddpg_swimmer"
    args.track = True
    args.save_model = True
    args.total_timesteps = 500000

for seed in [0,1,2]:
    args.seed = seed
    for mppi in [True, False]:
        args.mppi = mppi
        if mppi:
            for horizon in [1,3]:
                args.horizon = horizon
                for mu_in_mppi in [True, False]:
                    args.mu_in_mppi = mu_in_mppi
                    run_name = train(args)
        else:
            run_name = train(args)
