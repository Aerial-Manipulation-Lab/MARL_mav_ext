# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm, CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class UniformPoseCommandGlobal(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandGlobalCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandGlobalCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        # -- goal reached flag
        self.time_threshold_steps = int(2/env.sim.get_physics_dt()) # 2 seconds
        self.goal_dist_counter = torch.zeros(self.num_envs, device=self.device)
        self.achieved_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # used in reward function

    def __str__(self) -> str:
        msg = "UniformPoseCommandGlobal:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3] - self._env.scene.env_origins,
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.achieved_goal[env_ids] = False

    def _update_command(self):
        # Check if stable within goal
        within_goal_range = self.metrics["position_error"] < 0.2 and self.metrics["orientation_error"] < 0.2

        # Increment counter for environments within goal distance, reset to 0 for others
        self.goal_dist_counter = torch.where(
            within_goal_range,
            self.goal_dist_counter + 1,
            torch.zeros_like(self.goal_dist_counter)
        )

        # Set achieved goal if counter meets or exceeds the threshold
        self.achieved_goal |= self.goal_dist_counter >= self.time_threshold_steps
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3] + self._env.scene.env_origins, self.pose_command_w[:, 3:])
        # print("The tracking error of the position is ", self.metrics["position_error"])
        # print("The tracking error of the orientation is ", self.metrics["orientation_error"])

        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class UniformPoseCommandGlobalCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommandGlobal

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""


class RefTrajectoryCommand(CommandTerm):
    """Command generator for generating pose commands from a reference trajectory with time based sampling.

    The command generator generates poses by sampling positions and orientations from a reference trajectory
    given by the user. The reference trajectory is a sequence of poses (x, y, z, qw, qx, qy, qz) and twist (vx, vy, vz, wx, wy, wz).

    The sampling method is based on the time based sampler implemented in the Agilicious framework.
    """

    cfg: RefTrajectoryCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandGlobalCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # parameters for the trajectory
        self.reference_buffer = cfg.reference_trajectories
        _, num_setpoints, num_dimensions = self.reference_buffer.shape
        self.reference = torch.zeros(self.num_envs, num_setpoints, num_dimensions, device=self.device)
        self.num_points = cfg.num_points
        self.time_horizon = cfg.time_horizon

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.sim_dt = self._env.sim.get_rendering_dt()  # TODO: rendering dt has to be the same as planner dt
        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, self.num_points, 7, device=self.device)
        self.pose_command_w[..., 3] = 1.0
        self.twist_command = torch.zeros(self.num_envs, self.num_points, 6, device=self.device)
        self.acc_command = torch.zeros(self.num_envs, self.num_points, 6, device=self.device)
        self.sim_time = torch.zeros(self.num_envs, device=self.device)

        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["linear_velocity_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_velocity_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["linear_acceleration_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angular_acceleration_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommandGlobal:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose + twist command. Shape is (num_envs, num_points, 13).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        The next six elements correspond to the linear and angular velocities.
        """
        return torch.cat((self.pose_command_w, self.twist_command, self.acc_command), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error between the payload and the first point in the trajectory
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, 0, :3],
            self.pose_command_w[:, 0, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3] - self._env.scene.env_origins,
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        # compute the velocity error
        self.metrics["linear_velocity_error"] = torch.norm(
            self.twist_command[:, 0, :3] - self.robot.data.body_state_w[:, self.body_idx, 7:10], dim=-1
        )
        self.metrics["angular_velocity_error"] = torch.norm(
            self.twist_command[:, 0, 3:] - self.robot.data.body_state_w[:, self.body_idx, 10:], dim=-1
        )
        # compute the acceleration error
        self.metrics["linear_acceleration_error"] = torch.norm(
            self.acc_command[:, 0, :3] - self.robot.data.body_acc_w[:, self.body_idx, 0:3], dim=-1
        )
        self.metrics["angular_acceleration_error"] = torch.norm(
            self.acc_command[:, 0, 3:] - self.robot.data.body_acc_w[:, self.body_idx, 3:], dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # NOTE: Assigning new random reference trajectory and resetting to the starting point of that reference happens in the EventManager
        # sample random sim time for resetted envs between 0 and the max time of the reference trajectory - 10 seconds
        if self.cfg.random_init:
            self.sim_time[env_ids] = torch.rand_like(self.sim_time[env_ids], device=self.device) * (
                self.reference[0, -1, 0] - 10.0
            )
        else:
            self.sim_time[env_ids] = torch.zeros_like(self.sim_time[env_ids])

    def _update_command(self):
        """Update the sim time of each env and do time based sampling of the reference trajectory."""
        # get the time range of the reference trajectory
        if self.num_points > 1:
            timestamps = self.sim_time.unsqueeze(1) + torch.arange(self.num_points, device=self.device) / (
                (self.num_points - 1) / self.time_horizon
            )
            timestamps = torch.clamp(timestamps, max=self.reference[:, -2, 0].unsqueeze(1))

            setpoints = self.reference[:, :, 0].unsqueeze(1) > timestamps.unsqueeze(
                -1
            )  # Shape: (num_envs, num_points, num_setpoints)
            # Compute a boolean mask indicating which reference setpoints are greater than timestamps

            # Find the indices of the first setpoint greater than each timestamp
            setpoint_idxs = torch.argmax(setpoints.float(), dim=-1)  # Shape: (num_envs, num_points)

            # Gather the actions corresponding to these indices
            actions = torch.gather(
                self.reference,  # Shape: (num_envs, num_setpoints, action_dim)
                dim=1,
                index=setpoint_idxs.unsqueeze(-1).expand(
                    -1, -1, self.reference.shape[-1]
                ),  # Shape: (num_envs, num_points, action_dim)
            )  # Shape: (num_envs, num_points, action_dim)

        else:
            # get the time range of the reference trajectory
            setpoints = self.reference[:, :, 0] > self.sim_time.unsqueeze(1)
            setpoint_idx = torch.argmax(setpoints.float(), dim=1)
            actions = self.reference[:, setpoint_idx.data[0]].unsqueeze(1)

        # get the time based sampling
        pose = actions[..., 1:8]
        twist = actions[..., 8:14]
        acc = actions[..., 14:20]
        # update the command
        self.pose_command_w[:] = pose
        self.twist_command[:] = twist
        self.acc_command[:] = acc

        # update the sim time
        self.sim_time += self.sim_dt

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        marker_idx = [0] * self.num_points * self.num_envs
        positions = self.pose_command_w[..., :3] + self._env.scene.env_origins.unsqueeze(1)
        self.goal_pose_visualizer.visualize(
            positions.view(-1, 3), self.pose_command_w[..., 3:7].view(-1, 4), marker_indices=marker_idx
        )
        # print("The tracking error of the position is ", self.metrics["position_error"])
        # print("The tracking error of the orientation is ", self.metrics["orientation_error"])

        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class RefTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = RefTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    reference_trajectories: torch.Tensor = MISSING
    """Reference trajectory buffer to sample the trajectories from."""
    num_points: int = MISSING
    """Number of points in the reference trajectory."""
    time_horizon: float = MISSING
    """Time horizon of the number of points sampled from the reference trajectory."""
    random_init = True
    """Whether to randomly initialize drones along the reference trajectory or not."""
