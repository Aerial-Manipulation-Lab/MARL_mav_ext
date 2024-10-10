import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_inv, quat_mul

from .utils import get_drone_pdist, get_drone_rpos

"""
Observations for the payload
"""

# TODO: create helper file with functions for retrieving states


def payload_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload pose xyz, quat in env frame."""
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


def payload_linear_velocities(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_state_w[:, payload_idx, 7:10].view(env.num_envs, -1)


def payload_angular_velocities(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_state_w[:, payload_idx, 10:].view(env.num_envs, -1)

def payload_linear_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_acc_w[:, payload_idx, 0:3].view(env.num_envs, -1)

def payload_angular_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    return robot.data.body_acc_w[:, payload_idx, 3:].view(env.num_envs, -1)


def payload_positional_error(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload position error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_pos = env.command_manager.get_command("pose_command")[..., :3]
    positional_error = desired_pos - payload_pos_env
    return positional_error


def payload_orientation_error(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload orientation error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    desired_quat = env.command_manager.get_command("pose_command")[..., 3:]
    orientation_error = payload_quat - desired_quat
    return orientation_error


def cable_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angle of cable between cable and payload."""
    robot: Articulation = env.scene[asset_cfg.name]
    base_rope_idx = robot.find_bodies("rope_.*_link_0")[0]
    payload_idx = robot.find_bodies("load_link")[0]
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_state_w[:, payload_idx, 3:7].view(-1, 4).repeat(3, 1)
    payload_orientation_inv = quat_inv(payload_orientation_world)
    rope_orientations_payload = quat_mul(
        payload_orientation_inv, rope_orientations_world
    )  # cable angles relative to payload
    return rope_orientations_payload.view(env.num_envs, -1)


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

def drone_linear_acceleration(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_acc_w[:, drone_idx, 0:3].view(env.num_envs, -1)

def drone_angular_acceleration(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_acc_w[:, drone_idx, 3:].view(env.num_envs, -1)

# relative drone positions


def payload_drone_rpos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the payload from the drone."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    payload_idx = robot.find_bodies("load_link")[0]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    payload_pos_world_frame = robot.data.body_state_w[:, payload_idx, :3]
    rpos = drone_pos_world_frame - payload_pos_world_frame
    return rpos.view(env.num_envs, -1)


def drone_rpos_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the drones from each other."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    return rpos.view(env.num_envs, -1)


def drone_pdist_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Euclidean distance between drones."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = torch.norm(rpos, dim=-1, keepdim=True)
    return pdist.view(env.num_envs, -1)
