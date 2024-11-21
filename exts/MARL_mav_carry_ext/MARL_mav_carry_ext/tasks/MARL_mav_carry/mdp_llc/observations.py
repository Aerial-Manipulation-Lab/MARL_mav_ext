import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import quat_conjugate, quat_inv, quat_mul

from .utils import get_drone_pdist, get_drone_rpos

"""
Observations for the payload
"""

# Body indices found in the scene
# payload_idx = [0]
# drone_idx = [71, 72, 73]
# base_rope_idx = [8, 9, 10]

# for the case when the rod is used
payload_idx = [0]
drone_idx = [17, 18, 19]
base_rope_idx = [8, 9, 10]

def payload_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload pose xyz, quat in env frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_world_frame = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_env_frame = payload_world_frame - env.scene.env_origins
    return payload_env_frame.view(env.num_envs, -1)


def payload_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, payload_idx, 3:7].view(env.num_envs, -1)


def payload_linear_velocities(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, payload_idx, 7:10].view(env.num_envs, -1)


def payload_angular_velocities(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, payload_idx, 10:].view(env.num_envs, -1)


def payload_linear_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w[:, payload_idx, 0:3].view(env.num_envs, -1)


def payload_angular_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w[:, payload_idx, 3:].view(env.num_envs, -1)


def payload_positional_error(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload position error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins
    desired_pos = env.command_manager.get_command(command_name)[..., :3]
    positional_error = desired_pos - payload_pos_env
    return positional_error


def payload_orientation_error(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload orientation error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    orientation_error = quat_mul(desired_quat, quat_conjugate(payload_quat))
    return orientation_error


def payload_linear_velocity_error(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_state_w[:, payload_idx, 7:10].squeeze(1)
    desired_lin_vel = env.command_manager.get_command(command_name)[..., 7:10]
    lin_vel_error = desired_lin_vel - payload_lin_vel

    return lin_vel_error


def payload_angular_velocity_error(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity error."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:13].squeeze(1)
    desired_ang_vel = env.command_manager.get_command(command_name)[..., 10:13]
    ang_vel_error = desired_ang_vel - payload_ang_vel

    return ang_vel_error


def cable_angle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angle of cable between cable and payload."""
    robot: Articulation = env.scene[asset_cfg.name]
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_state_w[:, payload_idx, 3:7].repeat(1, 3, 1).view(-1, 4)
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
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    drone_pos_env_frame = drone_pos_world_frame - env.scene.env_origins.unsqueeze(1)
    return drone_pos_env_frame.view(env.num_envs, -1)


def drone_orientations(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone orientation, quaternions in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, drone_idx, 3:7].view(env.num_envs, -1)


def drone_linear_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone linear velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, drone_idx, 7:10].view(env.num_envs, -1)


def drone_angular_velocities(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone angular velocity in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_state_w[:, drone_idx, 10:].view(env.num_envs, -1)


def drone_linear_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone linear acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w[:, drone_idx, 0:3].view(env.num_envs, -1)


def drone_angular_acceleration(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Drone angular acceleration in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.body_acc_w[:, drone_idx, 3:].view(env.num_envs, -1)


# relative drone positions


def payload_drone_rpos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the payload from the drone."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    payload_pos_world_frame = robot.data.body_state_w[:, payload_idx, :3]
    rpos = drone_pos_world_frame - payload_pos_world_frame
    return rpos.view(env.num_envs, -1)


def drone_rpos_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Relative position of the drones from each other."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    return rpos.view(env.num_envs, -1)


def drone_pdist_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Euclidean distance between drones."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = torch.norm(rpos, dim=-1, keepdim=True)
    return pdist.view(env.num_envs, -1)

# Observations for when sampling multiple points on a trajectory

def payload_positional_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload position error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3]
    payload_pos_env = payload_pos_world - env.scene.env_origins.unsqueeze(1)
    desired_pos = env.command_manager.get_command(command_name)[..., :3]
    positional_error = (desired_pos - payload_pos_env).view(env.num_envs, -1)
    return positional_error


def payload_orientation_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload orientation error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    desired_quat = env.command_manager.get_command(command_name)[..., 3:7]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].repeat(1, desired_quat.shape[1], 1)
    orientation_error = quat_mul(desired_quat.view(-1, 4), quat_conjugate(payload_quat.view(-1, 4))).view(env.num_envs, -1)
    return orientation_error


def payload_linear_velocity_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear velocity error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_vel = robot.data.body_state_w[:, payload_idx, 7:10]
    desired_lin_vel = env.command_manager.get_command(command_name)[..., 7:10]
    lin_vel_error = (desired_lin_vel - payload_lin_vel).view(env.num_envs, -1)
    return lin_vel_error


def payload_angular_velocity_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular velocity error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:13]
    desired_ang_vel = env.command_manager.get_command(command_name)[..., 10:13]
    ang_vel_error = (desired_ang_vel - payload_ang_vel).view(env.num_envs, -1)
    return ang_vel_error

def payload_linear_acc_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload linear acceleration error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_lin_acc = robot.data.body_acc_w[:, payload_idx, 0:3]
    desired_lin_acc = env.command_manager.get_command(command_name)[..., 13:16]
    lin_acc_error = (desired_lin_acc - payload_lin_acc).view(env.num_envs, -1)
    return lin_acc_error

def payload_angular_acc_error_traj(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Payload angular acceleration error between the payload and all sampled points."""
    robot: Articulation = env.scene[asset_cfg.name]
    payload_ang_acc = robot.data.body_acc_w[:, payload_idx, 3:6]
    desired_ang_acc = env.command_manager.get_command(command_name)[..., 3:6]
    ang_acc_error = (desired_ang_acc - payload_ang_acc).view(env.num_envs, -1)
    return ang_acc_error