from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .utils import *
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# TODO: put this somewhere else
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


def track_payload_pos(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins

    # desired_pos = env.command_manager.get_command(command_name)[..., :3]  # relative goal generated in robot root frame.
    desired_pos = torch.zeros_like(payload_pos_env)
    desired_pos[..., 2] = 1.5  # in env frame
    # compute the error
    positional_error = torch.norm(desired_pos - payload_pos_env, dim=-1)
    reward_distance_scale= 1.2
    reward_position = torch.exp(-positional_error * reward_distance_scale)

    if env.scene.num_envs > 1:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
    else:
        marker_indices = [0, 1]

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


def track_payload_orientation(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload orientation commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_quat = env.command_manager.get_command(command_name)[..., 3:]  # 1 0 0 0
    # compute the error
    orientation_error = torch.norm(desired_quat - payload_quat, dim=-1)
    reward_distance_scale= 1.2
    reward_orientation = torch.exp(-orientation_error * reward_distance_scale)

    if env.scene.num_envs > 1:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
    else:
        marker_indices = [0, 1]

    if debug_vis:
        payload_orientation_marker.set_visibility(True)
        orientations = torch.cat((desired_quat, payload_quat), dim=0)
        desired_pos = torch.zeros_like(payload_pos_env)
        desired_pos[..., 2] = 1.5  # in env frame
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat((desired_pos_world, payload_pos_world), dim=0)
        payload_orientation_marker.visualize(positions, orientations, marker_indices=marker_indices)

    assert reward_orientation.shape == (env.scene.num_envs,)
    
    return reward_orientation


def separation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Test reward function."""
    safe_distance = 0.44 # smallest distance where drones are just upright
    robot = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = torch.norm(rpos, dim=-1, keepdim = True)
    separation = get_drone_pdist(pdist).min(dim=-1).values.min(dim=-1).values # get the smallest distance between drones in the swarm
    reward_separation = torch.square(separation / safe_distance).clamp(0,1)

    assert reward_separation.shape == (env.scene.num_envs,)
    return separation

def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for keeping the payload up."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_orientation = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_up = quat_axis(payload_orientation, axis=2)
    up = payload_up[:, 2]
    reward_up = torch.square((up + 1) / 2)
    assert reward_up.shape == (env.scene.num_envs,)
    return reward_up

def spinnage_reward_payload(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the payload."""
    spinnage_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_angular_velocity = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1).abs().sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(payload_angular_velocity))
    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin

def spinnage_reward_drones(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the angular velocities of the drones."""
    spinnage_weight = 0.4
    robot = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_angular_velocity = robot.data.body_state_w[:, drone_idx, 10:].square().sum(-1).sum(-1)
    reward_spin = spinnage_weight * torch.exp(-torch.square(drone_angular_velocity))
    assert reward_spin.shape == (env.scene.num_envs,)
    return reward_spin

def swing_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for minimizing the linear velocities of the payload."""
    swing_weight = 0.8
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_linear_velocity = robot.data.body_state_w[:, payload_idx, 7:10].squeeze(1).abs().sum(-1)
    reward_swing = swing_weight * torch.exp(-torch.square(payload_linear_velocity))
    assert reward_swing.shape == (env.scene.num_envs,)
    return reward_swing

def action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high action values."""
    reward_effort_weight= 0.2
    effort_norm = torch.norm(env.action_manager.action, dim=-1)
    reward_effort = reward_effort_weight * torch.exp(-effort_norm)
    assert reward_effort.shape == (env.scene.num_envs,)

    return reward_effort

def action_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for high action values."""
    reward_action_smoothness_weight = 0.2
    action_smoothness = torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=-1)
    reward_action_smoothness = reward_action_smoothness_weight * torch.exp(-action_smoothness)
    assert reward_action_smoothness.shape == (env.scene.num_envs,)
    return reward_action_smoothness

""" TODO: rewards for:
- Keeping the swarm in a certain separation distance
- Minimize angular velocities of payload (spinnage)
- Minimize linear velocities of payload (swing)
- Joint limits (angles between cables) of cable joints
- Action smoothness: penalize the difference between consecutive actions
"""

def OmniDrones_reward(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Rewards in the same way as the OmniDrones paper.
    This function calls the other functions to calculate the rewards,
    it is done in a seperate function because the seperation reward is a multiplicative factor.
    """

    # Calculate the rewards
    reward_position = track_payload_pos(env, debug_vis, command_name, asset_cfg)
    reward_orientation = track_payload_orientation(env, debug_vis, command_name, asset_cfg)
    reward_pose = reward_position + reward_orientation
    
    reward_separation = separation_reward(env)
    reward_up = upright_reward(env)
    reward_spin_payload = spinnage_reward_payload(env)
    reward_swing = swing_reward(env)

    reward_effort = action_penalty(env)
    reward_spin_drones = spinnage_reward_drones(env)

    # Calculate the total reward
    reward = reward_separation * (
                reward_pose
                + reward_pose * (reward_up + reward_spin_payload + reward_swing)
                # + reward_joint_limit
                # + reward_action_smoothness.mean(1, True) # set to 0 in omnidrones
                + reward_effort
                + reward_spin_drones
            )
    return reward