from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value_with_discount"
    args.track = True
    args.save_model = True
    args.var = 0.1
    args.mppi = True
    args.env_in_mppi = False
    args.Q_in_mppi = False
    args.mppi_targets = False
    args.target_horizon = 4 # unused, but for wandb filters
    args.num_target_rollouts = 20 # unused, but for wandb filters
    args.transition_network = "medium"
    args.total_timesteps = 500000
    args.transition_ensemble_size = None
    args.env_id = "Hopper-v5"
    # for seed in range(5):
    for seed in range(5,10):
        args.seed = seed
        for num_rollouts in [400,200]:
            args.num_rollouts = num_rollouts
            for horizon in [8,4]:
                args.horizon = horizon
                train(args)