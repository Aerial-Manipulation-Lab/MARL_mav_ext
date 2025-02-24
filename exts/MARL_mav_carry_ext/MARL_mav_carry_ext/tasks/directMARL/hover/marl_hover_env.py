# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .marl_hover_env_cfg import MARLHoverEnvCfg
from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
import MARL_mav_carry_ext.tasks.managerbased.mdp_llc as mdp


class MARLHoverEnv(DirectMARLEnv):
    cfg: MARLHoverEnvCfg

    def __init__(self, cfg: MARLHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # body indices
        self._falcon_idx = self.robot.find_bodies(cfg.falcon_names)[0]
        self._falcon_rotor_idx = self.robot.find_bodies(cfg.falcon_rotor_names)[0]
        self._payload_idx = self.robot.find_bodies(cfg.payload_name)[0]

        # configuration
        self._num_drones = len(self._falcon_idx)
        self._control_mode = cfg.control_mode

        # action buffers
        # buffers
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._prev_forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        self._setpoints = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}

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
        initial_rpms = [torch.tensor([[1880.4148, 1675.0350, 1670.4458, 1875.0309]], device=self.device).repeat(self.num_envs, 1),
                        torch.tensor([[1702.9099, 1894.7073, 1838.3457, 1636.0341]], device=self.device).repeat(self.num_envs, 1),
                        torch.tensor([[1337.6145, 1373.3019, 1519.6875, 1483.1881]], device=self.device).repeat(self.num_envs, 1)]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # termination buffers
        self._drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._load_position = torch.zeros(self.num_envs, 3, device=self.device)


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False) # TODO: not sure what this does
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        self._prev_forces = self._forces.clone()
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

            drone_positions = self.scene["robot"].data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
            drone_orientations = self.scene["robot"].data.body_com_state_w[:, self._falcon_idx, 3:7]
            drone_linear_velocities = self.scene["robot"].data.body_com_state_w[:, self._falcon_idx, 7:10]
            drone_angular_velocities = self.scene["robot"].data.body_com_state_w[:, self._falcon_idx, 10:13]
            drone_linear_accelerations = self.scene["robot"].data.body_acc_w[:, self._falcon_idx, :3]
            drone_angular_accelerations = self.scene["robot"].data.body_acc_w[:, self._falcon_idx, 3:6]

            for i in range(self._num_drones):
                drone_states: dict = {}  # dict of tensors
                drone_states["pos"] = drone_positions[:, i]
                drone_states["quat"] = drone_orientations[:, i]
                drone_states["lin_vel"] = drone_linear_velocities[:, i]
                drone_states["ang_vel"] = drone_angular_velocities[:, i]
                drone_states["lin_acc"] = drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = drone_angular_accelerations[:, i]
                # calculate current jerk and snap
                # self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self._sim_dt)
                # drone_states["jerk"] = self._drone_jerk[:, i]
                # self._drone_prev_acc[:, i] = drone_states["lin_acc"]
                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                )
                target_rpm = self._indi_controllers[i].getCommand(drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load)

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
        self.scene["robot"].set_external_force_and_torque(
            torch.zeros_like(self._moments), self._moments, self._falcon_idx
        )
        # apply forces to each rotor
        self.scene["robot"].set_external_force_and_torque(
            self._forces, torch.zeros_like(self._forces), self._falcon_rotor_idx
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "falcon1": torch.cat(
                (
                    torch.tensor([0.0], device=self.device),
                ),
                dim=-1,
            ),
            "falcon2": torch.cat(
                (
                   torch.tensor([0.0], device=self.device),
                ),
                dim=-1,
            ),
            "falcon3": torch.cat(
                (
                    torch.tensor([0.0], device=self.device),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                torch.tensor([0.0], device=self.device),
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # global rewards
        pos_rew = torch.tensor([0.0], device=self.device)
        ori_rew = torch.tensor([0.0], device=self.device)

        # rewards per drone
        action_smoothness_rew_1 = torch.tensor([0.0], device=self.device)
        action_rew_1 = torch.tensor([0.0], device=self.device)
        downwash_rew_1 = torch.tensor([0.0], device=self.device)

        action_smoothness_rew_2 = torch.tensor([0.0], device=self.device)
        action_rew_2 = torch.tensor([0.0], device=self.device)
        downwash_rew_2 = torch.tensor([0.0], device=self.device)

        action_smoothness_rew_3 = torch.tensor([0.0], device=self.device)
        action_rew_3 = torch.tensor([0.0], device=self.device)
        downwash_rew_3 = torch.tensor([0.0], device=self.device)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        # self.extras["log"]["dist_reward"] = rew_dist.mean()
        # self.extras["log"]["dist_goal"] = goal_dist.mean()

        return {"falcon1": pos_rew + ori_rew + action_smoothness_rew_1 + action_rew_1 + downwash_rew_1, 
                "falcon2": pos_rew + ori_rew + action_smoothness_rew_2 + action_rew_2 + downwash_rew_2,
                "falcon3": pos_rew + ori_rew + action_smoothness_rew_3 + action_rew_3 + downwash_rew_3}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        # crashing into ground
        # falcon_fly_low = self._drone_positions[:, :, 2] < 0.1
        payload_fly_low = self._load_position[:, 2] < 0.1
        print("payload_fly_low", payload_fly_low)

        # # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        

        terminated = {agent: payload_fly_low for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        # terminated = {"falcon1": torch.tensor([False], device=self.device),
        #                 "falcon2": torch.tensor([False], device=self.device),
        #                 "falcon3": torch.tensor([False], device=self.device)}

        return terminated, time_outs

    # def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
    #     if env_ids is None:
    #         env_ids = self.robot._ALL_INDICES
    #     # reset articulation and rigid body attributes
    #     super()._reset_idx(env_ids)

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        pass

    def _compute_intermediate_values(self):
        # data for falcons
        self._drone_positions[:] = self.scene["robot"].data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        self._load_position[:] = self.scene["robot"].data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins

        # # data for load
        # self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        # self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        # self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
        #     self.num_envs, self.num_fingertips, 3
        # )
        # self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        # self.left_hand_dof_pos = self.left_hand.data.joint_pos
        # self.left_hand_dof_vel = self.left_hand.data.joint_vel

        pass


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
