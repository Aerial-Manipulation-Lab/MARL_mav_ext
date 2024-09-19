import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat

"""
Observations for the payload
"""


def payload_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload pose xyz, quat in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_state_w[:, payload_idx, :7]  # xyz, quat


# def cable_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Cable angle between drone and payload."""
#     robot: Articulation = env.scene[asset_cfg.name]
#     #TODO
#     return pass

"""
Observations for the drones
"""


def drone_positions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone positions xyz in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_state_w[:, drone_idx, :3].view(env.num_envs, -1)

def drone_orientations(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone orientation, rpy in world frame."""
    # TODO: Check if view reshapes the quats in a logical way
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    quats = robot.data.body_state_w[:, drone_idx, 3:7]
    roll, pitch, yaw = euler_xyz_from_quat(quats.view(-1, 4))
    return torch.stack((roll, pitch, yaw), dim=1).view(env.num_envs, -1)

def drone_linear_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_state_w[:, drone_idx, 7:10].view(env.num_envs, -1)

def drone_angular_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_state_w[:, drone_idx, 10:].view(env.num_envs, -1)

# """
# Commands
# """
# def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     """The generated command from command term in the command manager with the given name."""
#     return env.command_manager.get_command(command_name)
