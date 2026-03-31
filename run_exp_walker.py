# from ddpg_continuous_action import Args, train
from ddpg_continuous_action_v2 import Args, train

if __name__ == "__main__":
    args = Args()
    args.wandb_entity = "mpcritic-dpc"
    args.wandb_project_name = "dual_mpcritic_ddpg_hopper_tb"
    args.track = True
    args.save_model = True
    args.env_in_mppi = False
    args.env_id = "Hopper-v5"

    # agreed upon
    args.total_timesteps = 500000
    args.lambda_ = 0.1
    args.transition_network = "medium"
    args.transition_ensemble_size = None
    args.mppi_handle_terminations = False
    args.trainer_scaler = 'null'
    args.distributional_dynamics = False

    args.mppi_prior = 'gaussian'
    args.batch_size = 32
    args.horizon = 4
    args.num_rollouts = 600 # 200 (previous experiments) or 500 (30% slower)
    args.num_target_rollouts = int(args.num_rollouts // 10)
    args.var = 0.15 # 0.1 (previous experiments) or 0.05 (seems to actually train on walker)
    args.double_Q = False
    args.learning_rate = 3e-4
    # args.episodic_learning = True
    args.target_horizon = 4
    for seed in range(5):
        for target_horizon in [args.horizon]:
            for lambda_ in [0.1, 0.05]:
                args.lambda_ = lambda_
                for training_pattern in ['online']:
                    for prior in ['uniform']:
                        args.mppi_prior = prior
                        args.target_horizon = target_horizon
                        args.seed = seed
                        args.training_pattern = training_pattern
                        train(args)