from soft_mpcritic import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = ""
    args.wandb_project_name = "soft_mpcritic"
    args.track = True
    args.save_model = True
    envs = ['InvertedDoublePendulum-v5', 'Hopper-v5']

    args.mppi_prior = 'gaussian'
    args.batch_size = 32
    args.horizon = 4
    args.transition_ensemble_size = None
    args.num_rollouts = 200
    args.lambda_ = 0.1
    args.var = 0.1

    # default uses mppi with Gaussian prior; 
    # uniform should use a smaller variance and we find needs more rollouts
    # try 600 for better stability but longer runtime
    # args.mppi_prior = "uniform"
    # args.num_rollouts = 600
    # args.var = 0.05
    # args.lambda_ = 0.15

    args.num_target_rollouts = int(args.num_rollouts // 10)

    for seed in range(1):
        for env in envs:
            if env == 'InvertedDoublePendulum-v5':
                args.total_timesteps = 50000
            elif env == 'Hopper-v5':
                args.total_timesteps = 500000
            args.env_id = env
            args.seed = seed
            train(args)