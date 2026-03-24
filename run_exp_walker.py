from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value_with_discount_walker"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Walker2d-v5"

    # agreed upon
    args.total_timesteps = 500000
    args.batch_size = 64
    args.horizon = 3
    args.target_horizon = 3
    args.num_rollouts = 200 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.1 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)

    # lambda experiments
    args.transition_network = "medium"
    args.transition_ensemble_size = None
    # for seed in range(5):
    #     args.seed = seed
    #     for lambda_ in [0.2, 0.5]:
    #         args.lambda_ = lambda_
    #         train(args)
    # for seed in range(2):
    #     args.seed = seed
    #     for lambda_ in [0.2]:
    #         args.lambda_ = lambda_
    #         for var in [0.4, 0.2, 0.05]:
    #             args.var = var
    #             train(args)

    # ensemble experiments
    args.lambda_ = 0.1
    # for seed in range(5):
    #     args.seed = seed
    #     train(args)

    # args.transition_network = "flex"
    # args.transition_ensemble_size = None
    # args.num_hidden_list = [2]*args.horizon
    # args.num_nodes_list = [64, 256, 512]
    # args.activations_list = [['silu', 'silu']]*args.horizon
    # for seed in range(5):
    #     args.seed = seed
    #     train(args)

    args.num_rollouts = 5000
    for seed in range(5):
        args.seed = seed
        for var in [0.4, 0.2, 0.1]:
            args.var = var
            train(args)