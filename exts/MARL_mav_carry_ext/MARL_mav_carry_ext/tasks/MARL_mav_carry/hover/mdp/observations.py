import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg

"""
Observations for the payload
"""


def payload_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload pose xyz, quat in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_world_frame = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_env_frame = payload_world_frame - env.scene.env_origins
    return payload_env_frame.view(env.num_envs, -1)


def payload_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_state_w[:, payload_idx, 3:7].view(env.num_envs, -1)


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
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    drone_pos_env_frame = drone_pos_world_frame - env.scene.env_origins.unsqueeze(1)
    return drone_pos_env_frame.view(env.num_envs, -1)


def drone_orientations(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_state_w[:, drone_idx, 3:7].view(env.num_envs, -1)


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
