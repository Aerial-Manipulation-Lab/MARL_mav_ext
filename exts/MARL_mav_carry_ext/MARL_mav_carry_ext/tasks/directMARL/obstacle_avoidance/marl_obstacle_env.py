# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections.abc import Sequence

from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import get_drone_pdist, get_drone_rpos

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer, DelayBuffer
from isaaclab.utils.math import (
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_unique,
    sample_uniform,
    sample_gaussian,
)

from .marl_obstacle_env_cfg import MARLObstacleEnvCfg


class MARLObstacleEnv(DirectMARLEnv):
    cfg: MARLObstacleEnvCfg

    def __init__(self, cfg: MARLObstacleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # body indices
        self._falcon_idx = self.robot.find_bodies(cfg.falcon_names)[0]
        self._falcon_rotor_idx = self.robot.find_bodies(cfg.falcon_rotor_names)[0]
        self._payload_idx = self.robot.find_bodies(cfg.payload_name)[0]
        self._rope_idx = self.robot.find_bodies(cfg.rope_name)[0]

        # configuration
        self._num_drones = len(self._falcon_idx)
        self._control_mode = cfg.control_mode

        # # observation buffers
        # self._observation_buffers = {}
        # for agent in self.cfg.possible_agents:
        #     self._observation_buffers[agent] = CircularBuffer(cfg.history_len, self.num_envs, device=self.device)
        # self._state_buffer = CircularBuffer(cfg.history_len, self.num_envs, device=self.device)

        # action buffers
        # buffers
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        # self.setpoint_delay_buffers = {}
        # for agent in self.cfg.possible_agents:
        #     self.setpoint_delay_buffers[agent] = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        #     self.setpoint_delay_buffers[agent].set_time_lag(cfg.constant_delay)
        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 12, device=self.device)
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 5, device=self.device)

        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.drone_orientations[..., 0] = 1.0
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        # obstacle terms
        self.wall_rpos_1 = torch.zeros(self.num_envs, self._num_drones + 1, 3, device=self.device) # TODO: remove other drones 
        self.wall_rpos_2 = torch.zeros(self.num_envs, self._num_drones + 1, 3, device=self.device)
        self.wall_size_1 = torch.zeros(self.num_envs, 3, device=self.device)
        self.wall_size_2 = torch.zeros(self.num_envs, 3, device=self.device)

        # curriculum terms
        self.vertical_curriculum = torch.full((self.num_envs,), self.cfg.curriculum_dist_max, device=self.device)
        self.horizontal_curriculum = torch.full((self.num_envs,), self.cfg.curriculum_dist_max, device=self.device)
        self.problem_choice = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.pos_reward_weight = torch.full((self.num_envs,), self.cfg.init_pos_track_weight, device=self.device)
        self.smoothness_weight = torch.full((self.num_envs,), self.cfg.init_smoothness_weight, device=self.device)
        self.stalling_seconds = torch.full((self.num_envs,), self.cfg.max_stalling_s, device=self.device)

        # max field size
        self.max_pos_error_norm = torch.norm(torch.tensor([cfg.bbox_distance, cfg.bbox_distance, cfg.bbox_distance / 2], device=self.device))

        # outer loop controller
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(self.num_envs, self._control_mode)
        self._ll_counter = 0
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)

        # inner loop controller
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # motor model
        # experimentally obtained
        self.motor_models = {}
        initial_rpms = [
            torch.tensor([[1441.5819, 1351.1626, 1341.0111, 1428.5597]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1377.9199, 1451.8428, 1408.9022, 1329.2014]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1281.3964, 1293.0708, 1361.7539, 1347.2434]], device=self.device).repeat(self.num_envs, 1),
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # load buffers
        self.load_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_orientation = torch.zeros(self.num_envs, 4, device=self.device)
        self.load_orientation[:, 0] = 1.0
        self.current_load_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.load_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_length_x = torch.tensor([[0.275, 0, 0]] * self.num_envs, device=self.device)
        self.load_length_y = torch.tensor([[0, 0.275, 0]] * self.num_envs, device=self.device)

        # Goal terms
        # # goal buffers
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0

        self.goal_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.difference_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)

        self.goal_dist_counter = torch.zeros(self.num_envs, device=self.device)
        self.achieved_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # used in reward function

        # reward logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pos_reward",
                "linear_pos_reward",
                "ori_reward",
                "action_smoothness",
                "body_rate_penalty",
                "force_penalty",
                "downwash_reward",
                # "goal_achieved_reward"
            ]
        }

        # # -- metrics
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # termination buffers
        self.falcon_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.payload_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.illegal_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_drone = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_load = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.cable_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.drone_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.body_pos_outside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.stalled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # debug vis
        self.set_debug_vis(cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        contact_sensors = ContactSensor(self.cfg.contact_forces)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # walls
        wall_1 = RigidObject(self.cfg.wall_1_cfg)
        wall_2 = RigidObject(self.cfg.wall_2_cfg)
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)  # TODO: not sure what this does
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_forces"] = contact_sensors
        self.scene.rigid_objects["wall_1"] = wall_1
        self.scene.rigid_objects["wall_2"] = wall_2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.cfg.possible_agents:
            # terms used for smoothness reward, current and previously calculated actions
            self.prev_actions[agent][:] = self.actions[agent]
            self.actions[agent][:] = actions[agent]

            # introduce delay in the setpoints
            # actions[agent][:] = self.setpoint_delay_buffers[agent].compute(actions[agent])

        for drone, action in actions.items():

            if self._control_mode == "geometric":
                self._setpoints[drone]["pos"] = action[:, :3]
                self._setpoints[drone]["lin_vel"] = action[:, 3:6]
                self._setpoints[drone]["lin_acc"] = action[:, 6:9]
                self._setpoints[drone]["jerk"] = action[:, 9:12]

                # self._desired_position[:, i] = self.drone_setpoint[i]["pos"]

            elif self._control_mode == "ACCBR":
                self._setpoints[drone]["lin_acc"] = action[:, :3]
                self._setpoints[drone]["body_rates"] = torch.cat((action[:, 3:], self._constant_yaw), dim=-1)

            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = self._constant_yaw
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            drone_positions = self.robot.data.body_com_state_w[
                :, self._falcon_idx, :3
            ] - self.scene.env_origins.unsqueeze(1)
            drone_orientations = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
            drone_linear_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
            drone_angular_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]
            drone_linear_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, :3]
            drone_angular_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, 3:6]

            self.drone_positions[:] = drone_positions  # + torch.randn_like(drone_positions) * self.position_noise_std
            self.drone_orientations[:] = (
                drone_orientations  # + torch.randn_like(drone_orientations) * self.orientation_noise_std
            )
            self.drone_linear_velocities[:] = (
                drone_linear_velocities  # + torch.randn_like(drone_linear_velocities) * self.linear_velocity_noise_std
            )
            self.drone_angular_velocities[:] = (
                drone_angular_velocities  # + torch.randn_like(drone_angular_velocities) * self.angular_velocity_noise_std
            )
            self.drone_linear_accelerations[:] = (
                drone_linear_accelerations  # + torch.randn_like(drone_linear_accelerations) * self.linear_acceleration_noise_std
            )
            self.drone_angular_accelerations[:] = (
                drone_angular_accelerations  # + torch.randn_like(drone_angular_accelerations) * self.angular_acceleration_noise_std
            )

            for i in range(self._num_drones):
                drone_states: dict = {}  # dict of tensors
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # calculate current jerk
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self.step_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                )
                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )

                # if self.cfg.debug_vis:
                #     self.drone_positions_debug[:, i] = drone_states["pos"] + self._env.scene.env_origins
                #     if self._control_mode == "geometric":
                #         self.drone_goals_debug[:, i] = self.drone_setpoint[i]["pos"] + self._env.scene.env_origins
                #     self.des_acc_debug[:, i] = acc_cmd
                #     self.des_ori_debug[:, i] = q_cmd

                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1

        # apply torques induced by rotors to each body
        self.robot.set_external_force_and_torque(torch.zeros_like(self._moments), self._moments, self._falcon_idx)
        # apply forces to each rotor
        self.robot.set_external_force_and_torque(self._forces, torch.zeros_like(self._forces), self._falcon_rotor_idx)

    def _get_observations(self) -> dict[str, torch.Tensor]:

        # local observations include:
        # load state
        # ego-drone drone
        # other-drone_states
        # goal terms
        # obstacle terms

        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )
        self.current_load_matrix[:] = matrix_from_quat(self.load_orientation)
        self.load_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 7:10].squeeze(1)
        self.load_ang_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 10:13].squeeze(1)

        self.drone_positions[:] = self.robot.data.body_com_state_w[
            :, self._falcon_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        self.drone_orientations[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        self.drone_linear_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
        self.drone_angular_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]

        self.goal_pos_error[:] = self.pose_command_w[:, :3] - self.load_position
        goal_load_matrix = matrix_from_quat(self.pose_command_w[:, 3:7])
        self.difference_matrix[:] = torch.matmul(goal_load_matrix, self.current_load_matrix.transpose(1, 2))

        wall_1_pos = self.scene["wall_1"].data.body_com_state_w[:, 0, :3] - self.scene.env_origins
        wall_2_pos = self.scene["wall_2"].data.body_com_state_w[:, 0, :3] - self.scene.env_origins
        self.wall_rpos_1[:, 0, :] = wall_1_pos - self.load_position
        self.wall_rpos_1[:, 1:, :] = wall_1_pos.unsqueeze(1) - self.drone_positions
        self.wall_rpos_2[:, 0, :] = wall_2_pos - self.load_position
        self.wall_rpos_2[:, 1:, :] = wall_2_pos.unsqueeze(1) - self.drone_positions

        # action_histories = []
        # for agent in self.cfg.possible_agents:
        #     if self.setpoint_delay_buffers[agent]._circular_buffer._buffer is None:
        #         action_history = self.actions[agent].unsqueeze(1).repeat(1, self.cfg.max_delay + 1, 1)
        #         action_histories.append(action_history)
        #     else:
        #         action_histories.append(self.setpoint_delay_buffers[agent]._circular_buffer.buffer)
        # self.all_action_histories = torch.cat(action_histories, dim=-1)

        obs_falcon1 = torch.cat(
            (
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                torch.tensor([[1, 0, 0]] * self.num_envs, device=self.device),  # one-hot encoding
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
                self.wall_rpos_1.view(self.num_envs, -1),
                self.wall_size_1,
                self.wall_rpos_2.view(self.num_envs, -1),
                self.wall_size_2,
            ),
            dim=-1,
        )

        # self._observation_buffers["falcon1"].append(obs_falcon1)

        obs_falcon2 = torch.cat(
            (
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                torch.tensor([[0, 1, 0]] * self.num_envs, device=self.device),  # one-hot encoding
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
                self.wall_rpos_1.view(self.num_envs, -1),
                self.wall_size_1,
                self.wall_rpos_2.view(self.num_envs, -1),
                self.wall_size_2,
            ),
            dim=-1,
        )

        # self._observation_buffers["falcon2"].append(obs_falcon2)

        obs_falcon3 = torch.cat(
            (
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                torch.tensor([[0, 0, 1]] * self.num_envs, device=self.device),  # one-hot encoding
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
                self.wall_rpos_1.view(self.num_envs, -1),
                self.wall_size_1,
                self.wall_rpos_2.view(self.num_envs, -1),
                self.wall_size_2,
            ),
            dim=-1,
        )

        # self._observation_buffers["falcon3"].append(obs_falcon3)

        # observations = {
        #     "falcon1": self._observation_buffers["falcon1"].buffer.reshape(self.num_envs, -1),
        #     "falcon2": self._observation_buffers["falcon2"].buffer.reshape(self.num_envs, -1),
        #     "falcon3": self._observation_buffers["falcon3"].buffer.reshape(self.num_envs, -1),
        # }
        observations = {
            "falcon1": obs_falcon1,
            "falcon2": obs_falcon2,
            "falcon3": obs_falcon3,
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # load terms
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                # goal terms
                self.goal_pos_error,
                self.difference_matrix.view(self.num_envs, -1),
                # self.all_action_histories.reshape(self.num_envs, -1),
                self.wall_rpos_1.view(self.num_envs, -1),
                self.wall_size_1,
                self.wall_rpos_2.view(self.num_envs, -1),
                self.wall_size_2,
            ),
            dim=-1,
        )

        # self._state_buffer.append(states)
        # return self._state_buffer.buffer.reshape(self.num_envs, -1)

        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # pos reward
        goal_pos_error_norm = torch.norm(self.pose_command_w[:, :3] - self.load_position, dim=-1)
        reward_distance_scale = 1.5
        reward_position = (
            self.pos_reward_weight * torch.exp(-goal_pos_error_norm * reward_distance_scale) * self.step_dt
        )

        relative_error = goal_pos_error_norm / self.max_pos_error_norm
        linear_pos_reward = self.pos_reward_weight * (1 - relative_error) * self.step_dt
        # orientation reward
        orientation_error = quat_error_magnitude(self.pose_command_w[:, 3:7], self.load_orientation)
        reward_distance_scale = 1.5
        reward_orientation = (
            self.cfg.ori_track_weight * torch.exp(-orientation_error * reward_distance_scale) * self.step_dt
        )

        # action smoothness reward
        current_actions = torch.cat([self.actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        action_prev = torch.cat([self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        diff_action = ((current_actions - action_prev).abs()) / self._num_drones
        reward_action_smoothness = (
            self.smoothness_weight * torch.exp(-torch.norm(diff_action, dim=-1).square()) * self.step_dt
        )

        # commanded body rate reward
        commanded_body_rates = torch.cat([self.actions[agent][:, 3:] for agent in self.cfg.possible_agents], dim=-1)
        body_rate_penalty = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
        reward_body_rate_penalty = self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_penalty) * self.step_dt

        # force penalty
        normalized_forces = self._forces[..., 2] / self.cfg.max_thrust_pp
        effort_sum = torch.max(normalized_forces, dim=-1)[0]
        reward_effort = self.cfg.force_penalty_weight * torch.exp(-effort_sum) * self.step_dt

        # downwash reward
        reward_downwash = self.cfg.downwash_rew_weight * self._downwash_reward() * self.step_dt

        # goal achieved reward
        # reward_goal_achieved = self.cfg.goal_achieved_bonus * self.achieved_goal.float()

        rewards = {
            "pos_reward": reward_position,
            "linear_pos_reward": linear_pos_reward,
            "ori_reward": reward_orientation,
            "action_smoothness": reward_action_smoothness,
            "body_rate_penalty": reward_body_rate_penalty,
            "force_penalty": reward_effort,
            "downwash_reward": reward_downwash,
            # "goal_achieved_reward": reward_goal_achieved,
        }

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        shared_rewards = (
            reward_position
            + linear_pos_reward
            + reward_orientation
            + reward_action_smoothness
            + reward_body_rate_penalty
            + reward_effort
            + reward_downwash
            # + reward_goal_achieved
        )

        return {agent: shared_rewards for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The terminations for the environment. Since all of the agents are connected by the cables,
        if 1 agent terminates, terminate all agents.
        """
        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )

        # crashing into ground
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < 0.1).any(dim=-1)
        self.payload_fly_low = self.load_position[:, 2] < 0.1

        # illegal contact
        contact_sensor = self.scene.sensors[self.cfg.sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        # check if any contact force exceeds the threshold
        self.illegal_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self.cfg.sensor_cfg.body_ids], dim=-1), dim=1)[0]
            > self.cfg.contact_sensor_threshold,
            dim=1,
        )

        # angle limits
        rope_orientations_world = self.robot.data.body_com_state_w[:, self._rope_idx, 3:7].view(-1, 4)
        drone_orientation_world = self.drone_orientations.view(-1, 4)
        drone_orientation_inv = quat_inv(drone_orientation_world)
        rope_orientations_drones = quat_mul(
            drone_orientation_inv, rope_orientations_world
        )  # cable angles relative to drones
        roll_drone, pitch_drone, _ = euler_xyz_from_quat(rope_orientations_drones)  # yaw can be whatever
        mapped_angle_drone = torch.stack((torch.cos(roll_drone), torch.cos(pitch_drone)), dim=1)
        self.angle_limit_drone = (
            (mapped_angle_drone < self.cfg.cable_angle_limits_drone).any(dim=1).view(-1, 3).any(dim=1)
        )

        self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
        payload_orientation_world = self.load_orientation.repeat(1, 3, 1).view(-1, 4)
        payload_orientation_inv = quat_inv(payload_orientation_world)
        rope_orientations_payload = quat_mul(
            payload_orientation_inv, rope_orientations_world
        )  # cable angles relative to payload
        roll_load, pitch_load, _ = euler_xyz_from_quat(rope_orientations_payload)  # yaw can be whatever
        mapped_angle_load = torch.stack((torch.cos(roll_load), torch.cos(pitch_load)), dim=1)
        self.angle_limit_load = (
            (mapped_angle_load < self.cfg.cable_angle_limits_payload).any(dim=1).view(-1, 3).any(dim=1)
        )

        # cables colliding
        self.cable_collision = self._cable_collision(
            self.cfg.cable_collision_threshold, self.cfg.cable_collision_num_points
        )

        # drones colliding
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = (
            pdist.min(dim=-1).values.min(dim=-1).values
        )  # get the smallest distance between drones in the swarm
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # bounding box
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)

        # update metrics
        self._update_metrics()

        # goal achieved
        within_goal_range = (self.metrics["position_error"] < self.cfg.goal_achieved_range) & (self.metrics["orientation_error"] < self.cfg.goal_achieved_ori_range)
        # Increment counter for environments within goal distance, reset to 0 for others
        self.goal_dist_counter = torch.where(
            within_goal_range, self.goal_dist_counter + 1, torch.zeros_like(self.goal_dist_counter)
        )
        # Set achieved goal if counter meets or exceeds the threshold
        self.achieved_goal |= self.goal_dist_counter >= (self.cfg.goal_time_threshold / self.step_dt)

        # terminate if it is not close enough after a certain amount of time
        stalling_time_passed = self.episode_length_buf >= (self.stalling_seconds / self.step_dt)
        self.stalled = stalling_time_passed & (self.metrics["position_error"] > self.cfg.stalling_distance)

        # reset when episode ends
        terminations = (
            self.falcon_fly_low
            | self.payload_fly_low
            | self.illegal_contact
            | self.angle_limit_drone
            | self.angle_limit_load
            | self.cable_collision
            | self.drone_collision
            | self.body_pos_outside
            | self.stalled
        )
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        timed_outs = self.time_out # | self.achieved_goal

        terminated = {agent: terminations for agent in self.cfg.possible_agents}
        time_outs = {agent: timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)
        self._reset_target_pose(env_ids)
        self._reset_robot_state_uniform(env_ids)

        # for agent in self.cfg.possible_agents:
        # self.setpoint_delay_buffers[agent].reset(env_ids)
        # self._observation_buffers[agent].reset(env_ids)
        # self._state_buffer.reset(env_ids)

        # curriculum, if an env succeeds to finish the episode within the threshold, level up, otherwise level down
        problems = self.problem_choice[env_ids]
        achieved_mask = self.achieved_goal[env_ids]

        vertical_envs = (problems == 0) # 0 is vertical
        horizontal_envs = (problems == 1) # 1 is horizontal

        # Update vertical curriculum (only for envs assigned to vertical problems)
        if vertical_envs.any():  # Check if any vertical envs exist in this batch
            self.vertical_curriculum[env_ids[vertical_envs]] -= achieved_mask[vertical_envs] * self.cfg.curriculum_dist_step
            self.vertical_curriculum[env_ids[vertical_envs]] += (~achieved_mask[vertical_envs]) * self.cfg.curriculum_dist_step
            self.vertical_curriculum[:] = torch.clamp(
                self.vertical_curriculum, 
                self.cfg.curriculum_dist_min, 
                self.cfg.curriculum_dist_max
            )

        # Update horizontal curriculum (only for envs assigned to horizontal problems)
        if horizontal_envs.any():  # Check if any horizontal envs exist in this batch
            self.horizontal_curriculum[env_ids[horizontal_envs]] -= achieved_mask[horizontal_envs] * self.cfg.curriculum_dist_step
            self.horizontal_curriculum[env_ids[horizontal_envs]] += (~achieved_mask[horizontal_envs]) * self.cfg.curriculum_dist_step
            self.horizontal_curriculum[:] = torch.clamp(
                self.horizontal_curriculum, 
                self.cfg.curriculum_dist_min, 
                self.cfg.curriculum_dist_max
            )
        
        # update reward weights
        # TODO: change vertical_curriculum to average of both
        self.pos_reward_weight[env_ids] = self.cfg.init_pos_track_weight + ((self.cfg.curriculum_dist_max - self.vertical_curriculum[env_ids])/(self.cfg.curriculum_dist_max - self.cfg.curriculum_dist_min)) * (self.cfg.max_pos_track_weight - self.cfg.init_pos_track_weight)
        self.smoothness_weight[env_ids] = self.cfg.init_smoothness_weight + ((self.cfg.curriculum_dist_max - self.vertical_curriculum[env_ids])/(self.cfg.curriculum_dist_max - self.cfg.curriculum_dist_min)) * (self.cfg.max_smoothness_weight - self.cfg.init_smoothness_weight)
        self.stalling_seconds[env_ids] = self.cfg.max_stalling_s + ((self.cfg.curriculum_dist_max - self.vertical_curriculum[env_ids])/(self.cfg.curriculum_dist_max - self.cfg.curriculum_dist_min)) * (self.cfg.min_stalling_s - self.cfg.max_stalling_s)
        # select new problem
        # self.problem_choice[env_ids.int()] = torch.randint(0, 2, size=(len(env_ids),), device=self.device, dtype=torch.int32)      

        # reset wall positions
        self._reset_wall_pos(env_ids, "wall_1")
        self._reset_wall_pos(env_ids, "wall_2")

        # reset goal achieved
        self.achieved_goal[env_ids] = False
        
        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["Episode_Termination/angle_drones_cable"] = torch.count_nonzero(
            self.angle_limit_drone[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/angle_load_cable"] = torch.count_nonzero(
            self.angle_limit_load[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cables_collide"] = torch.count_nonzero(
            self.cable_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/payload_fly_low"] = torch.count_nonzero(
            self.payload_fly_low[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/illegal_contact"] = torch.count_nonzero(
            self.illegal_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/stalled"] = torch.count_nonzero(
            self.stalled[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.time_out[env_ids]).item()

        self.extras["log"]["Curriculum/wall_dist_vertical"] = self.vertical_curriculum[env_ids].mean().item()

        self.extras["log"]["Curriculum/wall_dist_horizontal"] = self.horizontal_curriculum[env_ids].mean().item()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # log metrics
        for metric_name, metric_value in self.metrics.items():
            self.extras["log"][f"Metrics/pose_command/{metric_name}"] = metric_value.mean()

        # reset the action history
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.goal_range["pos_x"])
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.goal_range["pos_y"])
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.goal_range["pos_z"])
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.goal_range["roll"])
        euler_angles[:, 1].uniform_(*self.cfg.goal_range["pitch"])
        euler_angles[:, 2].uniform_(*self.cfg.goal_range["yaw"])
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique_command else quat

    def _reset_robot_state_uniform(self, env_ids):
        """Reset the robot state uniformly in the environment.
        50% of the time the sign of the goal and the robot x position are flipped."""
        root_states = self.robot.data.default_root_state[env_ids].clone()

        # poses
        range_list = [self.cfg.robot_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        # flip the sign of the goal and the robot x position
        flip_sign = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.int32)
        flip_sign = flip_sign * 2 - 1
        rand_samples[:, 0] = flip_sign * rand_samples[:, 0]
        self.pose_command_w[env_ids, 0] = flip_sign * self.pose_command_w[env_ids, 0]

        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

        # velocities, always 0
        velocities = torch.zeros_like(root_states[:, 7:13])

        # set into the physics simulation
        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    def _reset_wall_pos(self, env_ids, asset_name):
        """Reset the wall position based on the curriculum"""        
        wall = self.scene[asset_name]
        root_state = wall.data.default_root_state[env_ids].clone()
        
        # Initialize positions and orientations
        positions = torch.zeros_like(root_state[:, :3])
        orientations = torch.zeros_like(root_state[:, 3:7])
        curr_pos = torch.zeros(len(env_ids), 3, device=self.device)
        
        # Create masks
        new_vertical_mask = (self.problem_choice[env_ids] == 0)
        new_horizontal_mask = (self.problem_choice[env_ids] == 1)
        
        # Handle vertical environments
        if new_vertical_mask.any():
            vertical_ids = env_ids[new_vertical_mask]
            # gaussian distribution to sample around current curriculum
            curr_pos[new_vertical_mask, 1] = sample_gaussian(
                self.vertical_curriculum[vertical_ids],
                self.vertical_curriculum[vertical_ids]/3,
                (len(vertical_ids),),
                device=self.device,
            )
            
            if asset_name == "wall_1":
                positions[new_vertical_mask] = (root_state[new_vertical_mask, :3] + 
                                            self.scene.env_origins[vertical_ids] + 
                                            curr_pos[new_vertical_mask])
                self.wall_size_1[vertical_ids] = torch.tensor(self.cfg.wall_1_cfg.spawn.size, device=self.device)
            elif asset_name == "wall_2":
                positions[new_vertical_mask] = (root_state[new_vertical_mask, :3] + 
                                            self.scene.env_origins[vertical_ids] - 
                                            curr_pos[new_vertical_mask])
                self.wall_size_2[vertical_ids] = torch.tensor(self.cfg.wall_2_cfg.spawn.size, device=self.device)
            
            orientations[new_vertical_mask] = root_state[new_vertical_mask, 3:7]
        
        # Handle horizontal environments
        if new_horizontal_mask.any():
            horizontal_ids = env_ids[new_horizontal_mask]
            # gaussian distribution to sample around current curriculum
            curr_pos[new_horizontal_mask, 2] = sample_gaussian(
                self.horizontal_curriculum[horizontal_ids],
                self.horizontal_curriculum[horizontal_ids]/3,
                (len(horizontal_ids),),
                device=self.device,
            )

            # Create horizontal orientation quaternion (90° rotation around x-axis)
            new_orientation = torch.zeros_like(root_state[new_horizontal_mask, 3:7])
            new_orientation[:, 0] = 0.7071068
            new_orientation[:, 1] = 0.7071068
            orientations[new_horizontal_mask] = new_orientation
            
            if asset_name == "wall_1":
                offset = torch.tensor([0.0, -1.05, 0.5], device=self.device).expand(len(horizontal_ids), 3)
                positions[new_horizontal_mask] = (root_state[new_horizontal_mask, :3] + 
                                            self.scene.env_origins[horizontal_ids] + 
                                            offset + 
                                            curr_pos[new_horizontal_mask])
                self.wall_size_1[horizontal_ids] = torch.tensor(self.cfg.wall_1_cfg.spawn.size, device=self.device)
                self.wall_size_1[horizontal_ids] = quat_rotate(new_orientation, self.wall_size_1[horizontal_ids]).abs()
                
            elif asset_name == "wall_2":
                offset = torch.tensor([0.0, 1.05, -1.5], device=self.device).expand(len(horizontal_ids), 3)
                positions[new_horizontal_mask] = (root_state[new_horizontal_mask, :3] + 
                                            self.scene.env_origins[horizontal_ids] + 
                                            offset - 
                                            curr_pos[new_horizontal_mask])
                self.wall_size_2[horizontal_ids] = torch.tensor(self.cfg.wall_2_cfg.spawn.size, device=self.device)
                self.wall_size_2[horizontal_ids] = quat_rotate(new_orientation, self.wall_size_2[horizontal_ids]).abs()

        # Combine positions and orientations
        target_pose = torch.cat((positions, orientations), dim=-1)
        wall.write_root_pose_to_sim(target_pose, env_ids)

    def _update_metrics(self):
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.load_position,
            self.load_orientation,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _cable_collision(
        self,
        threshold: float = 0.0,
        num_points: int = 5,
    ) -> torch.Tensor:
        """Check for collisions between cables.

        A collision is detected if the minimum Euclidean distance between any two points
        on different cables is below the threshold.
        """
        cable_bottom_pos_env = self.robot.data.body_com_state_w[
            :, self._rope_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        cable_directions = self.drone_positions - cable_bottom_pos_env  # (num_envs, num_cables, 3)

        # Create linearly spaced points for interpolation (num_points,)
        linspace_points = torch.linspace(0, 1, num_points, device=self.device).view(
            1, 1, num_points, 1
        )  # (1, 1, num_points, 1)

        # Compute cable points (num_envs, num_cables, num_points, 3)
        cable_points = cable_bottom_pos_env.unsqueeze(2) + linspace_points * cable_directions.unsqueeze(
            2
        )  # (num_envs, num_cables, num_points, 3)

        # Flatten cable points for easier distance calculation (num_envs, num_cables * num_points, 3)
        cable_points_flat = cable_points.view(self.num_envs, -1, 3)

        # Pairwise distance calculation
        cable_points_a = cable_points_flat.unsqueeze(2)  # (num_envs, num_points_total, 1, 3)
        cable_points_b = cable_points_flat.unsqueeze(1)  # (num_envs, 1, num_points_total, 3)
        pairwise_diff = cable_points_a - cable_points_b  # (num_envs, num_points_total, num_points_total, 3)
        pairwise_distances = torch.norm(pairwise_diff, dim=-1)  # (num_envs, num_points_total, num_points_total)

        # Mask to ignore self-distances and distances within the same cable
        num_cables = cable_bottom_pos_env.shape[1]
        points_per_cable = num_points

        # Create mask to ignore points on the same cable
        cable_indices = torch.arange(num_cables, device=self.device).repeat_interleave(
            points_per_cable
        )  # (num_points_total,)
        same_cable_mask = cable_indices.unsqueeze(0) == cable_indices.unsqueeze(
            1
        )  # (num_points_total, num_points_total)
        same_cable_mask = same_cable_mask.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )  # (num_envs, num_points_total, num_points_total)

        # Apply mask: set ignored distances to a large value
        pairwise_distances[same_cable_mask] = 1000.0

        # Find the minimum distance across all points in each environment
        min_distances, _ = torch.min(pairwise_distances.view(self.num_envs, -1), dim=-1)  # Shape: (num_envs,)

        # Check if the minimum distance is below the threshold
        is_cable_collision = min_distances < threshold  # Shape: (num_envs,)

        assert is_cable_collision.shape == (self.num_envs,)
        return is_cable_collision

    def _downwash_reward(self):
        # Plane equation for the payload
        x_len_payload_env = quat_rotate(self.load_orientation, self.load_length_x)
        y_len_payload_env = quat_rotate(self.load_orientation, self.load_length_y)
        edge_payload_x = self.load_position + x_len_payload_env
        edge_payload_y = self.load_position + y_len_payload_env
        plane_vec1 = edge_payload_x - self.load_position
        plane_vec2 = edge_payload_y - self.load_position
        normal = torch.linalg.cross(plane_vec1, plane_vec2)
        d = torch.sum(normal * self.load_position, dim=-1).unsqueeze(-1).unsqueeze(-1)  # Shape (num_envs, 1, 1)

        # Line equations for each drone's thrust direction
        thrust_directions = quat_rotate(
            self.drone_orientations.view(-1, 4),
            torch.tensor([[0, 0, 1.0]] * self.num_envs * self._num_drones, device=self.device),
        ).view(self.num_envs, self._num_drones, 3)

        # Calculate intersection points with the plane
        # t = (d - normal * drone_pos) / (normal * thrust_direction)
        numerator = d - torch.sum(normal.unsqueeze(1) * self.drone_positions, dim=-1, keepdim=True)
        denominator = (
            torch.sum(normal.unsqueeze(1) * thrust_directions, dim=-1, keepdim=True) + 1e-6
        )  # Avoid division by zero
        t = numerator / denominator

        # Intersection points on the plane for each drone
        line_point_proj = self.drone_positions + t * thrust_directions  # Shape (num_envs, num_drones, 3)

        # Calculate distance between intersection points and payload position
        line_dist = torch.norm(
            line_point_proj - self.load_position.unsqueeze(1), dim=-1
        )  # Shape (num_envs, num_drones)
        # Reward: penalize based on distance from the intersection point to the payload position
        scaling_factor = 3
        reward_downwash = 1 - torch.exp(
            -torch.min(line_dist, dim=-1).values * scaling_factor
        )  # Min distance from the payload
        return reward_downwash

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not hasattr(self, "goal_pose_visualizer"):
            # -- goal pose
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_goal)
            # -- current body pose
            self.body_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_body)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3] + self.scene.env_origins, self.pose_command_w[:, 3:]
        )

        # -- current body pose
        self.body_pose_visualizer.visualize(self.load_position + self.scene.env_origins, self.load_orientation)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
