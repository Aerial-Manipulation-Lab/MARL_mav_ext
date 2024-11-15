from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_inv, quat_mul

from .marker_utils import DRONE_POS_MARKER_CFG
from .utils import *

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# TODO: put this somewhere else
num_drones = 3

# Body indices found in the scene
# payload_idx = [0]
# drone_idx = [71, 72, 73]
# base_rope_idx = [8, 9, 10]

# for the case when the rod is used
payload_idx = [0]
drone_idx = [17, 18, 19]
base_rope_idx = [8, 9, 10]

debug_vis_reward = True
if debug_vis_reward:
    GOAL_ORIENTATION_MARKER_CFG = VisualizationMarkersCfg(
        markers={
            "goal_frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
            "current_frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        }
    )

    orientation_marker_cfg = GOAL_ORIENTATION_MARKER_CFG.copy()
    orientation_marker_cfg.prim_path = "/Visuals/payload_orientation"
    payload_orientation_marker = VisualizationMarkers(orientation_marker_cfg)


def separation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Test reward function."""
    safe_distance = 0.44  # smallest distance where drones are just upright
    robot = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = get_drone_pdist(pdist)
    separation = pdist.min(dim=-1).values.min(dim=-1).values  # get the smallest distance between drones in the swarm
    reward_separation = torch.square(separation / safe_distance).clamp(0, 1)

    assert reward_separation.shape == (env.scene.num_envs,)
    return separation


def track_drone_reference(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for tracking the drone reference."""
    robot = env.scene[asset_cfg.name]
    drone_pos_world = robot.data.body_state_w[:, drone_idx, :3]
    drone_pos_env = drone_pos_world - env.scene.env_origins.unsqueeze(1)
    desired_pos = env.action_manager._terms["low_level_action"]._desired_position
    # compute the error
    positional_error = torch.norm((desired_pos - drone_pos_env) / num_drones, dim=-1)
    total_positional_error = positional_error.sum(dim=-1)
    sep_reward = separation_reward(env, asset_cfg)
    reward_distance_scale = 1.0
    reward_position = torch.exp(-total_positional_error * reward_distance_scale) * sep_reward
    assert reward_position.shape == (env.scene.num_envs,)
    return reward_position


def track_payload_pos_command(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins

    desired_pos = env.command_manager.get_command(command_name)[
        ..., :3
    ]  # relative goal generated in robot root frame, use a goal in env frame
    # compute the error

    # for the trajectory case
    if len(desired_pos.shape) > 2:
        desired_pos = desired_pos[:,0]

    positional_error = torch.norm(desired_pos - payload_pos_env, dim=-1)
    reward_distance_scale = 1.5
    reward_position = torch.exp(-positional_error * reward_distance_scale)

    assert reward_position.shape == (env.scene.num_envs,)

    return reward_position


def track_payload_orientation_command(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload orientation commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    # compute the error
    # for the trajectory case
    if len(desired_quat.shape) > 2:
        desired_quat = desired_quat[:,0]
        
    orientation_error = quat_error_magnitude(desired_quat, payload_quat)
    reward_distance_scale = 1.5
    reward_orientation = torch.exp(-orientation_error * reward_distance_scale)

    if debug_vis:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
        payload_orientation_marker.set_visibility(True)
        orientations = torch.cat((desired_quat, payload_quat), dim=0)
        desired_pos = env.command_manager.get_command(command_name)[
            ..., :3
        ]  # relative goal generated in robot root frame, use a goal in env frame
        
        # for the trajectory case
        if len(desired_pos.shape) > 2:
            desired_pos = desired_pos[:,0]
        
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat((desired_pos_world, payload_pos_world), dim=0)
        payload_orientation_marker.visualize(positions, orientations, marker_indices=marker_indices)

    assert reward_orientation.shape == (env.scene.num_envs,)

    return reward_orientation


def track_payload_pose_command(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for the desired pose"""
    position_reward = track_payload_pos_command(env, debug_vis_reward, command_name, asset_cfg)
    orientation_reward = track_payload_orientation_command(env, debug_vis_reward, command_name, asset_cfg)
    reward_pose = position_reward + orientation_reward

    assert reward_pose.shape == (env.scene.num_envs,)
    return reward_pose


def track_payload_lin_vel_command(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload linear velocity commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_state_w[:, payload_idx, 7:10].squeeze(1)

    desired_vel = env.command_manager.get_command(command_name)[..., 7:10]

    # for the trajectory case
    if len(desired_vel.shape) > 2:
        desired_vel = desired_vel[:,0]
        
    lin_vel_error = torch.norm(desired_vel - payload_lin_vel, dim=-1)
    reward_distance_scale = 15.0
    reward_lin_vel = torch.exp(-lin_vel_error * reward_distance_scale)

    assert reward_lin_vel.shape == (env.scene.num_envs,)

    return reward_lin_vel


def track_payload_ang_vel_command(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload angular velocity commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1)

    desired_vel = env.command_manager.get_command(command_name)[..., 10:]
    
    # for the trajectory case
    if len(desired_vel.shape) > 2:
        desired_vel = desired_vel[:,0]
        
    ang_vel_error = torch.norm(desired_vel - payload_ang_vel, dim=-1)
    reward_distance_scale = 15.0
    reward_ang_vel = torch.exp(-ang_vel_error * reward_distance_scale)

    assert reward_ang_vel.shape == (env.scene.num_envs,)

    return reward_ang_vel


def track_payload_twist_command(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for the desired twist"""
    lin_vel_reward = track_payload_lin_vel_command(env, debug_vis_reward, command_name, asset_cfg)
    ang_vel_reward = track_payload_ang_vel_command(env, debug_vis_reward, command_name, asset_cfg)
    reward_twist = lin_vel_reward + ang_vel_reward

    assert reward_twist.shape == (env.scene.num_envs,)
    return reward_twist


def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping the payload up."""
    robot = env.scene[asset_cfg.name]
    payload_orientation = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_up = quat_axis(payload_orientation, axis=2)
    up = payload_up[:, 2]
    reward_up = torch.square((up + 1) / 2)
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_up = reward_up * sep_reward * pose_reward  # from omnidrones paper

    assert reward_up.shape == (env.scene.num_envs,)
    return reward_up


def spinnage_reward_payload(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the payload."""
    spinnage_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_angular_velocity = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1).abs().sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(payload_angular_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_spin = reward_spin * sep_reward * pose_reward  # from omnidrones paper

    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def spinnage_reward_drones(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the drones."""
    spinnage_weight = 0.8
    robot = env.scene[asset_cfg.name]
    drone_angular_velocity = (robot.data.body_state_w[:, drone_idx, 10:] / num_drones).square().sum(-1).sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(drone_angular_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    reward_spin = reward_spin * sep_reward  # from omnidrones paper

    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def swing_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the linear velocities of the payload."""
    swing_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_linear_velocity = robot.data.body_state_w[:, payload_idx, 7:10].squeeze(1).abs().sum(-1)
    reward_swing = swing_weight * torch.exp(-torch.square(payload_linear_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_swing = reward_swing * sep_reward * pose_reward  # from omnidrones paper

    assert reward_swing.shape == (env.scene.num_envs,)
    return reward_swing


# if using spline:
def jerk_penalty_spline(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for desired jerk values."""
    reward_jerk_weight = 0.2
    desired_jerk = env.action_manager._terms["low_level_action"]._desired_jerk / num_drones
    jerk_norm_drone = torch.norm(desired_jerk, dim=-1)
    total_jerk = jerk_norm_drone.sum(dim=-1)
    reward_jerk = reward_jerk_weight * torch.exp(-total_jerk)
    sep_reward = separation_reward(env)
    reward_jerk = reward_jerk * sep_reward  # from omnidrones paper

    assert reward_jerk.shape == (env.scene.num_envs,)
    return reward_jerk


def snap_penalty_spline(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for desired snap values."""
    reward_snap_weight = 0.5
    desired_snap = env.action_manager._terms["low_level_action"]._desired_snap / num_drones
    snap_norm_drone = torch.norm(desired_snap, dim=-1)
    total_snap = snap_norm_drone.sum(dim=-1)
    reward_snap = reward_snap_weight * torch.exp(-total_snap)
    sep_reward = separation_reward(env)
    reward_snap = reward_snap * sep_reward  # from omnidrones paper

    assert reward_snap.shape == (env.scene.num_envs,)
    return reward_snap


def action_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high variation in action values."""
    action = env.action_manager.action
    action_prev = env.action_manager.prev_action
    action_smoothness = torch.norm((action - action_prev) / num_drones, dim=-1)
    scaling_factor = 5
    reward_action_smoothness = torch.exp(-action_smoothness * scaling_factor)

    assert reward_action_smoothness.shape == (env.scene.num_envs,)
    return reward_action_smoothness


def action_penalty_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high force values."""
    reward_effort_weight = 0.5
    action_forces = env.action_manager._terms["low_level_action"].processed_actions[..., 2]
    normalized_forces = action_forces / 6.25
    effort_sum = torch.sum(normalized_forces, dim=-1) / num_drones / 4  # num propellers
    reward_effort = reward_effort_weight * torch.exp(-effort_sum)
    reward_effort = reward_effort

    assert reward_effort.shape == (env.scene.num_envs,)
    return reward_effort


def action_penalty_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high force values."""
    action_forces = env.action_manager._terms["low_level_action"].processed_actions[..., 2]
    normalized_forces = action_forces / 6.25
    average_force_prop = torch.sum(normalized_forces, dim=-1) / num_drones / 4  # num propellers
    min_effort_prop = 0.5702
    effective_effort = torch.abs(average_force_prop - min_effort_prop)
    reward_effort = torch.exp(-effective_effort)

    assert reward_effort.shape == (env.scene.num_envs,)
    return reward_effort


def action_smoothness_force_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high variation in force values."""
    action_force = env.action_manager._terms["low_level_action"].processed_actions[..., 2] / 6.25
    action_prev_force = env.action_manager._terms["low_level_action"]._prev_forces[..., 2] / 6.25
    action_smoothness_force = torch.sum(action_force - action_prev_force, dim=-1) / num_drones / 4  # num propellors
    reward_action_smoothness_force = torch.exp(-action_smoothness_force)

    assert reward_action_smoothness_force.shape == (env.scene.num_envs,)
    return reward_action_smoothness_force


def angle_cable_load(
    env: ManagerBasedRLEnv, threshold: float = 0.261799388, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for desired angle between load and cables"""
    robot = env.scene[asset_cfg.name]
    reward_weight = 1.2
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_state_w[:, payload_idx, 3:7].repeat(1, 3, 1).view(-1, 4)
    payload_orientation_inv = quat_inv(payload_orientation_world)
    rope_orientations_payload = quat_mul(
        payload_orientation_inv, rope_orientations_world
    )  # cable angles relative to payload
    desired_angles = torch.tensor(
        [[-threshold, threshold], [threshold, threshold], [0, -threshold]], device=env.sim.device
    ).repeat(env.scene.num_envs, 1, 1)
    roll, pitch, yaw = euler_xyz_from_quat(rope_orientations_payload)  # yaw can be whatever
    mapped_angle = torch.stack((torch.sin(roll.view(env.num_envs, 3)), torch.sin(pitch.view(env.num_envs, 3))), dim=-1)
    angle_error = torch.norm(mapped_angle - desired_angles, dim=-1)
    reward_angle = reward_weight * torch.exp(-angle_error.sum(dim=-1) / num_drones)

    assert reward_angle.shape == (env.scene.num_envs,)
    return reward_angle


def downwash_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), debug_vis: bool = True
) -> torch.Tensor:
    """Reward for keeping the downwash wake away from the payload"""
    robot = env.scene[asset_cfg.name]
    reward_weight = 0.5

    # Plane equation for the payload
    payload_pose_env = robot.data.body_state_w[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    payload_orientation = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_length = torch.tensor([[0.275, 0.225, 0]] * env.num_envs, device=env.sim.device)
    len_payload_env = quat_rotate(payload_orientation, payload_length)
    edge_payload_min = payload_pose_env - len_payload_env
    edge_payload_max = payload_pose_env + len_payload_env
    plane_vec1 = edge_payload_max - payload_pose_env
    plane_vec2 = edge_payload_min - payload_pose_env
    normal = torch.linalg.cross(plane_vec1, plane_vec2)
    d = torch.sum(normal * payload_pose_env, dim=-1).unsqueeze(-1).unsqueeze(-1)  # Shape (num_envs, 1, 1)

    # Line equations for each drone's thrust direction
    drone_pos_env = robot.data.body_state_w[:, drone_idx, :3] - env.scene.env_origins.unsqueeze(1)
    drone_orientation = robot.data.body_state_w[:, drone_idx, 3:7].view(-1, 4)
    thrust_directions = quat_rotate(
        drone_orientation, torch.tensor([[0, 0, 1.0]] * env.num_envs * num_drones, device=env.sim.device)
    ).view(env.num_envs, num_drones, 3)

    # Calculate intersection points with the plane
    # t = (d - normal * drone_pos) / (normal * thrust_direction)
    numerator = d - torch.sum(normal.unsqueeze(1) * drone_pos_env, dim=-1, keepdim=True)
    denominator = (
        torch.sum(normal.unsqueeze(1) * thrust_directions, dim=-1, keepdim=True) + 1e-6
    )  # Avoid division by zero
    t = numerator / denominator

    # Intersection points on the plane for each drone
    line_point_proj = drone_pos_env + t * thrust_directions  # Shape (num_envs, num_drones, 3)

    # Calculate distance between intersection points and payload position
    line_dist = torch.norm(line_point_proj - payload_pose_env.unsqueeze(1), dim=-1)  # Shape (num_envs, num_drones)
    straight_line_dist = torch.tensor(
        [[1.1941, 1.1941, 1.1735]] * env.num_envs, device="cuda:0"
    )  # line dist when drones are straight under the load
    diff_line_dist = line_dist - straight_line_dist
    # Reward: penalize based on distance from the intersection point to the payload position
    reward_downwash = (
        reward_weight
        * torch.max(1 - torch.exp(-diff_line_dist * 15), torch.tensor(0.0, device=env.sim.device)).min(dim=-1)[0]
    )  # Min distance from the payload

    assert reward_downwash.shape == (env.num_envs,)
    return reward_downwash
