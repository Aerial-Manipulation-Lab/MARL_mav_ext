from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_error_magnitude, quat_inv, quat_mul, quat_rotate_inverse

from .marker_utils import DRONE_POS_MARKER_CFG
from .utils import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# TODO: put this somewhere else
num_drones = 3

# Body indices found in the scene
# payload_idx = [0]
# drone_idx = [71, 72, 73]
# base_rope_idx = [8, 9, 10]

# for the case when the rod is used
payload_idx = [0]
drone_idx = [20, 27, 34]
base_rope_idx = [8, 9, 10]

def separation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Separation reward function."""
    safe_distance = 0.44  # smallest distance where drones are just upright
    robot = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = get_drone_pdist(rpos)
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

def track_payload_pos_command_linear(
    env: ManagerBasedRLEnv, bbox: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands with linear kernel.
    Args:
        bbox: The bounding box half size for the linear kernel.
    """
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_com_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_pos = env.command_manager.get_command(command_name)[..., :3]
    # compute the error
    positional_error = torch.norm(desired_pos - payload_pos_env, dim=-1)
    max_error = torch.norm(torch.tensor([bbox, bbox, bbox/2], device=env.sim.device))
    relative_error = positional_error / max_error
    reward_distance_scale = 1.0 
    reward_position = 1 - relative_error * reward_distance_scale

    assert reward_position.shape == (env.scene.num_envs,)
    return reward_position

def track_payload_pos_command(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands with exponentional kernel."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_com_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins

    desired_pos = env.command_manager.get_command(command_name)[
        ..., :3
    ]  # relative goal generated in robot root frame, use a goal in env frame
    # compute the error

    # for the trajectory case
    if len(desired_pos.shape) > 2:
        desired_pos = desired_pos[:, 0]

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
    payload_quat = robot.data.body_com_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_pos_world = robot.data.body_com_state_w[:, payload_idx, :3].squeeze(1)
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    # compute the error
    # for the trajectory case
    if len(desired_quat.shape) > 2:
        desired_quat = desired_quat[:, 0]

    orientation_error = quat_error_magnitude(desired_quat, payload_quat)
    reward_distance_scale = 1.5
    reward_orientation = torch.exp(-orientation_error * reward_distance_scale)

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
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 15.0
) -> torch.Tensor:
    """Reward tracking of payload linear velocity commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_com_state_w[:, payload_idx, 7:10].squeeze(1)

    desired_vel = env.command_manager.get_command(command_name)[..., 7:10]

    # for the trajectory case
    if len(desired_vel.shape) > 2:
        desired_vel = desired_vel[:, 0]

    lin_vel_error = torch.norm(desired_vel - payload_lin_vel, dim=-1)
    reward_lin_vel = torch.exp(-lin_vel_error * scale)

    assert reward_lin_vel.shape == (env.scene.num_envs,)

    return reward_lin_vel


def track_payload_ang_vel_command(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 15.0
) -> torch.Tensor:
    """Reward tracking of payload angular velocity commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_com_state_w[:, payload_idx, 10:].squeeze(1)

    desired_vel = env.command_manager.get_command(command_name)[..., 10:13]

    # for the trajectory case
    if len(desired_vel.shape) > 2:
        desired_vel = desired_vel[:, 0]

    ang_vel_error = torch.norm(desired_vel - payload_ang_vel, dim=-1)
    reward_ang_vel = torch.exp(-ang_vel_error * scale)

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
    payload_orientation = robot.data.body_com_state_w[:, payload_idx, 3:7].squeeze(1)
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
    robot = env.scene[asset_cfg.name]
    payload_angular_velocity = robot.data.body_com_state_w[:, payload_idx, 10:].squeeze(1).abs().sum(-1)
    reward_spin = torch.exp(-torch.square(payload_angular_velocity))
    
    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def spinnage_reward_drones(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the drones."""
    robot = env.scene[asset_cfg.name]
    drone_angular_velocity_magnitude = torch.norm(robot.data.body_state_w[:, drone_idx, 10:], dim=-1)
    max_ang_vels = torch.max(drone_angular_velocity_magnitude, dim=-1)[0]
    reward_spin = torch.exp(-torch.square(max_ang_vels))

    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def swing_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the linear velocities of the payload."""
    robot = env.scene[asset_cfg.name]
    payload_linear_velocity = robot.data.body_com_state_w[:, payload_idx, 7:10].squeeze(1).abs().sum(-1)
    reward_swing = torch.exp(-torch.square(payload_linear_velocity))

    assert reward_swing.shape == (env.scene.num_envs,)
    return reward_swing

def action_policy_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high variation in action values."""
    action = env.action_manager.action
    action_prev = env.action_manager.prev_action
    action_smoothness = torch.norm((action - action_prev) / num_drones, dim=-1)
    reward_action_smoothness = torch.exp(-action_smoothness)

    assert reward_action_smoothness.shape == (env.scene.num_envs,)
    return reward_action_smoothness

def action_penalty_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high force values."""
    action_forces = env.action_manager._terms["low_level_action"].processed_actions[..., 2]
    normalized_forces = action_forces / 6.25
    effort_sum = torch.sum(normalized_forces, dim=-1) / num_drones / 4  # num propellers
    reward_effort = torch.exp(-effort_sum)
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
    action_force = env.action_manager._terms["low_level_action"].processed_actions[..., 2]
    action_prev_force = env.action_manager._terms["low_level_action"]._prev_forces[..., 2]
    diff_force = action_force - action_prev_force
    max_diff_force = torch.max(diff_force, dim=-1)[0]
    reward_action_smoothness_force = torch.exp(-max_diff_force * 5)

    assert reward_action_smoothness_force.shape == (env.scene.num_envs,)
    return reward_action_smoothness_force


def angle_cable_load(
    env: ManagerBasedRLEnv, threshold: float = 0.261799388, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for desired angle between load and cables"""
    robot = env.scene[asset_cfg.name]
    reward_weight = 1.2
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_com_state_w[:, payload_idx, 3:7].repeat(1, 3, 1).view(-1, 4)
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

    # Plane equation for the payload
    payload_pose_env = robot.data.body_com_state_w[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    payload_orientation = robot.data.body_com_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_length_x = torch.tensor([[0.275, 0, 0]] * env.num_envs, device=env.sim.device)
    payload_length_y = torch.tensor([[0, 0.275, 0]] * env.num_envs, device=env.sim.device)
    x_len_payload_env = quat_rotate(payload_orientation, payload_length_x)
    y_len_payload_env = quat_rotate(payload_orientation, payload_length_y)
    edge_payload_x = payload_pose_env + x_len_payload_env
    edge_payload_y = payload_pose_env + y_len_payload_env
    plane_vec1 = edge_payload_x - payload_pose_env
    plane_vec2 = edge_payload_y - payload_pose_env
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
    # Reward: penalize based on distance from the intersection point to the payload position
    scaling_factor = 3
    reward_downwash = (1 - torch.exp(-torch.min(line_dist, dim=-1).values * scaling_factor))  # Min distance from the payload

    assert reward_downwash.shape == (env.num_envs,)
    return reward_downwash

def obstacle_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), obstacle_cfg: SceneEntityCfg = SceneEntityCfg("wall")
) -> torch.Tensor:
    """Penalty for getting close to the obstacle."""
    robot = env.scene[asset_cfg.name]
    obstacle = env.scene[obstacle_cfg.name]
    payload_pos_env = robot.data.body_com_state_w[:, payload_idx, :3] - env.scene.env_origins.unsqueeze(1)
    drones_pos_env = robot.data.body_state_w[:, drone_idx, :3] - env.scene.env_origins.unsqueeze(1)
    obstacle_pos = obstacle.data.body_com_state_w[:, 0, :3].unsqueeze(1) - env.scene.env_origins.unsqueeze(1)
    all_bodies_env = torch.cat((payload_pos_env, drones_pos_env), dim=1)
    rpos = torch.abs(all_bodies_env - obstacle_pos)
    cuboid_dims = torch.tensor([[1.0, 1.75, 2.5]] * env.num_envs, device=env.sim.device).unsqueeze(1) # half lenghts
    # check if any of the bodies are inside the cuboid
    cuboid_dims_world = quat_rotate(obstacle.data.body_com_state_w[:, 0, 3:7].unsqueeze(1), cuboid_dims)
    is_inside_cuboid = torch.all(rpos <= cuboid_dims_world, dim=-1) # Shape (num_envs, num_bodies) true or false for each body
    reward_obstacle = -torch.any(is_inside_cuboid, dim=-1).float() # Shape (num_envs,) -1 if any body is inside the cuboid, 0 otherwise

    assert reward_obstacle.shape == (env.scene.num_envs,)
    return reward_obstacle

def goal_reached_reward(
    env: ManagerBasedRLEnv, command_name: str, 
) -> torch.Tensor:
    """Reward for reaching the goal and staying there, then terminate the episde."""
    reward_goal = env.command_manager._terms[command_name].achieved_goal

    assert reward_goal.shape == (env.scene.num_envs,)
    return reward_goal