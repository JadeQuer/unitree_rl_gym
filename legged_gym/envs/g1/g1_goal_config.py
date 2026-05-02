from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1GoalRoughCfg(G1RoughCfg):
    class env(G1RoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        num_envs = 2048
        episode_length_s = 24

    class commands(G1RoughCfg.commands):
        curriculum = False
        resampling_time = 0.2
        heading_command = True
        class ranges(G1RoughCfg.commands.ranges):
            lin_vel_x = [0.0, 1.0]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.5]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 0.8

    class rewards(G1RoughCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.5
        class scales(G1RoughCfg.rewards.scales):
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.7
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.8
            base_height = -8.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.2
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            goal_progress = 18.0
            goal_velocity = 2.0
            goal_reached = 40.0
            goal_heading = 1.0
            goal_distance = 2.0
            stand_still = -0.0


class G1GoalRoughCfgPPO(G1RoughCfgPPO):
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.4
        actor_hidden_dims = [64, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(G1RoughCfgPPO.runner):
        experiment_name = "g1_goal"
        max_iterations = 2500
