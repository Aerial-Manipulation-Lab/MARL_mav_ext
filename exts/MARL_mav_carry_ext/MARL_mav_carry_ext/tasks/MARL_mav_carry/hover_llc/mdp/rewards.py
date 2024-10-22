from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from .utils import *

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# TODO: put this somewhere else
num_drones = 3

debug_vis_reward = True
if debug_vis_reward:
    GOAL_POS_MARKER_CFG = VisualizationMarkersCfg(
        markers={
            "goal_pos_marker": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "current_pos_marker": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    pos_marker_cfg = GOAL_POS_MARKER_CFG.copy()
    pos_marker_cfg.prim_path = "/Visuals/payload_pos"
    payload_pos_marker = VisualizationMarkers(pos_marker_cfg)

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
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = torch.norm(rpos, dim=-1, keepdim=True)
    separation = (
        get_drone_pdist(pdist).min(dim=-1).values.min(dim=-1).values
    )  # get the smallest distance between drones in the swarm
    reward_separation = torch.square(separation / safe_distance).clamp(0, 1)

    assert reward_separation.shape == (env.scene.num_envs,)
    return separation


def track_payload_pos(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins

    desired_pos = env.command_manager.get_command(command_name)[
        ..., :3
    ]  # relative goal generated in robot root frame, use a goal in env frame
    # compute the error
    positional_error = torch.norm(desired_pos - payload_pos_env, dim=-1)
    reward_distance_scale = 1.2
    reward_position = torch.exp(-positional_error * reward_distance_scale)

    marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs

    if debug_vis:
        # set their visibility to true
        payload_pos_marker.set_visibility(True)
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat(
            (desired_pos_world, payload_pos_world), dim=0
        )  # visualize the payload positions in world frame
        payload_pos_marker.visualize(translations=positions, marker_indices=marker_indices)

    assert reward_position.shape == (env.scene.num_envs,)

    return reward_position

def track_drone_reference(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for tracking the drone reference."""
    robot = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_pos_world = robot.data.body_state_w[:, drone_idx, :3]
    drone_pos_env = drone_pos_world - env.scene.env_origins.unsqueeze(1)
    desired_pos = env.action_manager.action.view(-1,3,3)
    # compute the error
    positional_error = torch.norm((desired_pos - drone_pos_env) / num_drones, dim=-1)
    total_positional_error = positional_error.sum(dim=-1)
    reward_distance_scale = 1.5
    reward_position = torch.exp(-total_positional_error * reward_distance_scale)
    assert reward_position.shape == (env.scene.num_envs,)
    return reward_position



def track_payload_orientation(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload orientation commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    desired_quat = env.command_manager.get_command(command_name)[..., 3:]
    # compute the error
    orientation_error = torch.norm(desired_quat - payload_quat, dim=-1)
    reward_distance_scale = 1.2
    reward_orientation = torch.exp(-orientation_error * reward_distance_scale)

    if env.scene.num_envs > 1:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
    else:
        marker_indices = [0, 1]

    if debug_vis:
        payload_orientation_marker.set_visibility(True)
        orientations = torch.cat((desired_quat, payload_quat), dim=0)
        desired_pos = env.command_manager.get_command(command_name)[
            ..., :3
        ]  # relative goal generated in robot root frame, use a goal in env frame
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat((desired_pos_world, payload_pos_world), dim=0)
        payload_orientation_marker.visualize(positions, orientations, marker_indices=marker_indices)

    assert reward_orientation.shape == (env.scene.num_envs,)

    return reward_orientation

def track_payload_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for the desired pose"""
    position_reward = track_payload_pos(env, debug_vis_reward, "pose_command", asset_cfg)
    orientation_reward = track_payload_orientation(env, debug_vis_reward, "pose_command", asset_cfg)
    reward_pose = position_reward + orientation_reward
    sep_reward = separation_reward(env, asset_cfg)
    reward_pose = reward_pose * sep_reward # from omnidrones paper
    assert reward_pose.shape == (env.scene.num_envs,)
    return reward_pose

def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping the payload up."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_orientation = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_up = quat_axis(payload_orientation, axis=2)
    up = payload_up[:, 2]
    reward_up = torch.square((up + 1) / 2)
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_up = reward_up * sep_reward * pose_reward # from omnidrones paper

    assert reward_up.shape == (env.scene.num_envs,)
    return reward_up


def spinnage_reward_payload(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the payload."""
    spinnage_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_angular_velocity = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1).abs().sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(payload_angular_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_spin = reward_spin * sep_reward * pose_reward # from omnidrones paper

    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def spinnage_reward_drones(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the drones."""
    spinnage_weight = 0.8
    robot = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_angular_velocity = (robot.data.body_state_w[:, drone_idx, 10:] / num_drones).square().sum(-1).sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(drone_angular_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    reward_spin = reward_spin * sep_reward # from omnidrones paper

    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin


def swing_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the linear velocities of the payload."""
    swing_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_linear_velocity = robot.data.body_state_w[:, payload_idx, 7:10].squeeze(1).abs().sum(-1)
    reward_swing = swing_weight * torch.exp(-torch.square(payload_linear_velocity))
    sep_reward = separation_reward(env, asset_cfg)
    pose_reward = track_payload_pose(env, asset_cfg)
    reward_swing = reward_swing * sep_reward * pose_reward # from omnidrones paper

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
    reward_jerk = reward_jerk * sep_reward # from omnidrones paper

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
    reward_snap = reward_snap * sep_reward # from omnidrones paper

    assert reward_snap.shape == (env.scene.num_envs,)
    return reward_snap


def action_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high variation in action values."""
    reward_action_smoothness_weight = 0.5
    action = env.action_manager.action
    action_prev = env.action_manager.prev_action
    action_smoothness = torch.norm((action - action_prev) / num_drones, dim=-1)
    reward_action_smoothness = reward_action_smoothness_weight * torch.exp(-action_smoothness)
    sep_reward = separation_reward(env)
    reward_action_smoothness = reward_action_smoothness * sep_reward # from omnidrones paper

    assert reward_action_smoothness.shape == (env.scene.num_envs,)
    return reward_action_smoothness


def action_penalty_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high force values."""
    reward_effort_weight = 0.2
    action_forces = env.action_manager._terms["low_level_action"].processed_actions[..., 2]
    normalized_forces = action_forces / 6.25
    effort_norm = torch.sum(normalized_forces, dim=-1) / num_drones
    reward_effort = reward_effort_weight * torch.exp(-effort_norm)
    sep_reward = separation_reward(env)
    reward_effort = reward_effort * sep_reward # from omnidrones paper

    assert reward_effort.shape == (env.scene.num_envs,)
    return reward_effort


def action_smoothness_force_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high variation in force values."""
    reward_action_smoothness_force_weight = 0.5
    action_force = env.action_manager._terms["low_level_action"].processed_actions[..., 2] / 6.25
    action_prev_force = env.action_manager._terms["low_level_action"]._prev_forces[..., 2] / 6.25
    action_smoothness_force = torch.norm((action_force - action_prev_force) / num_drones, dim=-1)
    reward_action_smoothness_force = reward_action_smoothness_force_weight * torch.exp(-action_smoothness_force)
    sep_reward = separation_reward(env)
    reward_action_smoothness_force = reward_action_smoothness_force * sep_reward # from omnidrones paper

    assert reward_action_smoothness_force.shape == (env.scene.num_envs,)
    return reward_action_smoothness_force   

""" TODO: rewards for:
- Joint limits (angles between cables) of cable joints
"""