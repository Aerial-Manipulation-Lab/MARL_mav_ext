from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
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
            )
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
    payload_pos_world = (
        robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    )
    payload_pos_env = payload_pos_world - env.scene.env_origins

    # desired_pos = env.command_manager.get_command(command_name)[..., :3]  # relative goal generated in robot root frame.
    desired_pos = torch.zeros_like(payload_pos_env)
    desired_pos[..., 2] = 1.5 # in env frame
    # compute the error
    positional_error = torch.sum(
        torch.square(payload_pos_env - desired_pos),
        dim=1,
    )
    if env.scene.num_envs > 1:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
    else:
        marker_indices = [0, 1]

    if debug_vis:
        # set their visibility to true
        payload_pos_marker.set_visibility(True)
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat((desired_pos_world, payload_pos_world), dim=0) # visualize the payload positions in world frame
        payload_pos_marker.visualize(translations = positions, marker_indices = marker_indices)

    return -positional_error


def track_payload_orientation(
    env: ManagerBasedRLEnv, debug_vis: bool, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload orientation commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    payload_pos_world = (
        robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    )
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_quat = env.command_manager.get_command(command_name)[..., 3:] # 1 0 0 0
    # compute the error
    orientation_error = torch.sum(
        torch.abs(payload_quat - desired_quat),
        dim=1,
    )

    if env.scene.num_envs > 1:
        marker_indices = [0] * env.scene.num_envs + [1] * env.scene.num_envs
    else:
        marker_indices = [0, 1]

    if debug_vis:
        payload_orientation_marker.set_visibility(True)
        orientations = torch.cat((desired_quat, payload_quat), dim=0)
        desired_pos = torch.zeros_like(payload_pos_env)
        desired_pos[..., 2] = 1.5    # in env frame
        desired_pos_world = desired_pos + env.scene.env_origins
        positions = torch.cat((desired_pos_world, payload_pos_world), dim=0)
        payload_orientation_marker.visualize(positions, orientations, marker_indices = marker_indices)
    
    return -orientation_error

def action_penalty(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for high action values."""
    return -torch.sum(env.action_manager.action ** 2, dim=1)

""" TODO: rewards for:
- Keeping the swarm in a certain seperation distance
- Minimize angular velocities of payload (spinnage)
- Minimize linear velocities of payload (swing)
- Joint limits (angles between cables) of cable joints
- Action smoothness: penalize the difference between consecutive actions
"""
