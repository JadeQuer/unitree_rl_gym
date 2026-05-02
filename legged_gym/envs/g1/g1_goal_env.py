import torch

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *


class G1GoalRobot(G1Robot):
    def _init_buffers(self):
        super()._init_buffers()
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.goal_dir_world = torch.zeros((self.num_envs, 2), device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.needs_goal_resample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _sample_goals(self, env_ids):
        goal_x = torch_rand_float(1.0, 3.0, (len(env_ids), 1), device=self.device).squeeze(1)
        goal_y = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.goal_pos[env_ids, 0] = self.root_states[env_ids, 0] + goal_x
        self.goal_pos[env_ids, 1] = self.root_states[env_ids, 1] + goal_y
        self.goal_pos[env_ids, 2] = self.root_states[env_ids, 2]
        goal_vec = self.goal_pos[env_ids, :2] - self.root_states[env_ids, :2]
        goal_dist = torch.norm(goal_vec, dim=1)
        self.prev_goal_dist[env_ids] = goal_dist
        self.goal_dist[env_ids] = goal_dist
        self.goal_dir_world[env_ids] = goal_vec / goal_dist.clamp(min=1e-6).unsqueeze(1)
        self.goal_reached[env_ids] = False
        self.needs_goal_resample[env_ids] = False

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        self._sample_goals(env_ids)

    def _post_physics_step_callback(self):
        self.update_feet_state()

        resample_env_ids = self.needs_goal_resample.nonzero(as_tuple=False).flatten()
        if len(resample_env_ids) > 0:
            self._sample_goals(resample_env_ids)

        goal_vec_world = self.goal_pos[:, :2] - self.base_pos[:, :2]
        goal_dist = torch.norm(goal_vec_world, dim=1).clamp(min=1e-6)
        self.goal_dist = goal_dist
        goal_dir_world = goal_vec_world / goal_dist.unsqueeze(1)
        self.goal_dir_world = goal_dir_world

        yaw = torch.atan2(
            2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] + self.base_quat[:, 0] * self.base_quat[:, 1]),
            1.0 - 2.0 * (self.base_quat[:, 1] ** 2 + self.base_quat[:, 2] ** 2),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        goal_dir_body_x = cos_yaw * goal_dir_world[:, 0] + sin_yaw * goal_dir_world[:, 1]
        goal_dir_body_y = -sin_yaw * goal_dir_world[:, 0] + cos_yaw * goal_dir_world[:, 1]
        heading = torch.atan2(goal_dir_body_y, goal_dir_body_x)

        forward_speed = torch.clamp(goal_dist * 1.0, 0.15, 1.0)
        lateral_speed = torch.clamp(goal_dir_body_y * 0.4, -0.25, 0.25)
        yaw_rate = torch.clamp(heading * 1.8, -1.0, 1.0)

        self.commands[:, 0] = forward_speed * torch.cos(torch.atan2(goal_dir_body_y, goal_dir_body_x))
        self.commands[:, 1] = lateral_speed
        self.commands[:, 2] = yaw_rate
        self.commands[:, 3] = yaw

        self.goal_reached = goal_dist < 0.45
        self.needs_goal_resample |= self.goal_reached
        self.goal_progress = self.prev_goal_dist - goal_dist
        self.prev_goal_dist = goal_dist

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        LeggedRobot._post_physics_step_callback(self)

        self.commands[:, 0] = forward_speed
        self.commands[:, 1] = lateral_speed
        self.commands[:, 2] = yaw_rate
        self.commands[:, 3] = yaw

        return None

    def compute_observations(self):
        super().compute_observations()

    def _reward_goal_progress(self):
        return self.goal_progress

    def _reward_goal_reached(self):
        return self.goal_reached.float()

    def _reward_goal_distance(self):
        return torch.exp(-self.goal_dist)

    def _reward_goal_velocity(self):
        world_lin_vel = quat_apply(self.base_quat, self.base_lin_vel)[:, :2]
        return torch.clamp(torch.sum(world_lin_vel * self.goal_dir_world, dim=1), min=0.0)

    def _reward_goal_heading(self):
        goal_vec_world = self.goal_pos[:, :2] - self.base_pos[:, :2]
        goal_yaw = torch.atan2(goal_vec_world[:, 1], goal_vec_world[:, 0])
        heading = torch.atan2(
            2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] + self.base_quat[:, 0] * self.base_quat[:, 1]),
            1.0 - 2.0 * (self.base_quat[:, 1] ** 2 + self.base_quat[:, 2] ** 2),
        )
        heading_error = torch.abs(wrap_to_pi(goal_yaw - heading))
        return 1.0 - torch.clamp(heading_error / 1.57, 0.0, 1.0)
