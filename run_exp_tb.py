from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_warmstart_ensemble"
    args.env_id = "InvertedDoublePendulum-v5"
    args.track = True
    args.save_model = True
    args.total_timesteps = 50000
    args.var = 0.1
    args.mppi = True
    args.mppi_targets = True
    args.num_target_rollouts = 10
    args.env_in_mppi = False
    args.num_rollouts = 100
    args.transition_network = "medium"

    for seed in range(3):
        args.seed = seed+1
        for horizon in [4,8]:
            args.horizon = horizon
            for transition_ensemble_size in [None, 1]:
                args.transition_ensemble_size = transition_ensemble_size
                for mppi_target_warmstart in [True, False]:
                    args.mppi_target_warmstart = mppi_target_warmstart
                    run_name = train(args)
