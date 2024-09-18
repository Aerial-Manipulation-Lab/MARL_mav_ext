from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def track_payload_pos(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload position commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos = robot.data.body_state_w[:, payload_idx, :3]
    desired_pos = env.command_manager.get_command(command_name)[..., :3] # this is in the base frame of the robot
    # compute the error
    positional_error = torch.sum(
        torch.square(payload_pos - desired_pos),
        dim=1,
    )
    return positional_error.sum()

def track_payload_orientation(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of payload orientation commands."""
    robot: RigidObject = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7]
    desired_quat = env.command_manager.get_command(command_name)[..., 3:]
    # compute the error
    orientation_error = torch.sum(
        torch.abs(payload_quat - desired_quat),
        dim=1,
    )
    return orientation_error.sum()
