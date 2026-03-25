from ddpg_continuous_action import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_new_value_with_discount_iterations_soap"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Walker2d-v5"

    # agreed upon
    args.total_timesteps = 500000
    # args.batch_size = 64
    args.horizon = 2
    args.num_rollouts = 200 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.1 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    args.lambda_ = 0.1

    args.transition_network = "medium_deep"
    args.transition_ensemble_size = None
    args.model_optimizer = "soap"

    for seed in range(5):
        args.seed = seed
        for use_huber in [False, True]:
            args.use_huber_loss = use_huber
            for learning_rate in [3e-4]:
                for batch_size in [64]:
                    args.batch_size = batch_size
                    args.learning_rate = learning_rate
                    train(args)