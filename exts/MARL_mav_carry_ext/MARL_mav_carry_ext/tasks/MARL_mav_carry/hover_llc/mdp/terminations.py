import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_inv, quat_mul
from .utils import get_drone_pdist, get_drone_rpos

# Body indices found in the scene
payload_idx = [0]
drone_idx = [71, 72, 73]
base_rope_idx = [8, 9, 10]
top_rope_idx = [62, 63, 64]

def falcon_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the falcon flies too low."""
    robot = env.scene[asset_cfg.name]
    falcon_pos = robot.data.body_state_w[:, drone_idx, :3] - env.scene.env_origins.unsqueeze(1)
    is_falcon_pos_low = (falcon_pos[..., 2] < threshold).any(dim=1)
    assert is_falcon_pos_low.shape == (env.num_envs,)
    return is_falcon_pos_low


def falcon_spin(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the falcon spins too fast."""
    robot = env.scene[asset_cfg.name]
    falcon_ang_vel = robot.data.body_state_w[:, drone_idx, 10:].reshape(env.scene.num_envs, -1)
    is_falcon_spin = (falcon_ang_vel > threshold).any(dim=1)
    assert is_falcon_spin.shape == (env.num_envs,)
    return is_falcon_spin


def payload_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload flies too low."""
    robot = env.scene[asset_cfg.name]
    payload_pos = robot.data.body_state_w[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    is_payload_low = payload_pos[:, 2] < threshold
    assert is_payload_low.shape == (env.num_envs,)
    return is_payload_low


def payload_spin(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload spins too fast."""
    robot = env.scene[asset_cfg.name]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1)
    is_payload_spin = (payload_ang_vel > threshold).any(dim=1)
    assert is_payload_spin.shape == (env.num_envs,)
    return is_payload_spin


def payload_angle_cos(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload angle is too large."""
    robot = env.scene[asset_cfg.name]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    roll, pitch, yaw = euler_xyz_from_quat(payload_quat)  # yaw can be whatever
    mapped_angle = torch.stack((torch.cos(roll), torch.cos(pitch)), dim=1)
    is_angle_limit = (mapped_angle < threshold).any(dim=1)
    assert is_angle_limit.shape == (env.num_envs,)
    return is_angle_limit


def cable_angle_drones_cos(
    env: ManagerBasedRLEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle of cable between cable and drones."""
    robot = env.scene[asset_cfg.name]
    rope_orientations_world = robot.data.body_state_w[:, top_rope_idx, 3:7].view(-1, 4)
    drone_orientation_world = robot.data.body_state_w[:, drone_idx, 3:7].view(-1, 4)
    drone_orientation_inv = quat_inv(drone_orientation_world)
    rope_orientations_drones = quat_mul(
        drone_orientation_inv, rope_orientations_world
    )  # cable angles relative to drones
    roll, pitch, yaw = euler_xyz_from_quat(rope_orientations_drones)  # yaw can be whatever
    mapped_angle = torch.stack((torch.cos(roll), torch.cos(pitch)), dim=1)
    is_cable_limit = (mapped_angle < threshold).any(dim=1).view(-1, 3).any(dim=1)
    assert is_cable_limit.shape == (env.num_envs,)
    return is_cable_limit


def cable_angle_payload_cos(
    env: ManagerBasedRLEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle of cable between cable and payload."""
    robot: Articulation = env.scene[asset_cfg.name]
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
    payload_orientation_world = robot.data.body_state_w[:, payload_idx, 3:7].repeat(1, 3, 1).view(-1, 4)
    payload_orientation_inv = quat_inv(payload_orientation_world)
    rope_orientations_payload = quat_mul(
        payload_orientation_inv, rope_orientations_world
    )  # cable angles relative to payload
    roll, pitch, yaw = euler_xyz_from_quat(rope_orientations_payload)  # yaw can be whatever
    mapped_angle = torch.stack((torch.cos(roll), torch.cos(pitch)), dim=1)
    is_cable_limit = (mapped_angle < threshold).any(dim=1).view(-1, 3).any(dim=1)
    assert is_cable_limit.shape == (env.num_envs,)
    return is_cable_limit


def bounding_box(
    env: ManagerBasedRLEnv, threshold: float = 3.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload is outside the bounding box."""
    robot = env.scene[asset_cfg.name]
    body_pos = robot.data.body_state_w[:, payload_idx, :3]
    body_pos_env = body_pos - env.scene.env_origins.unsqueeze(1)
    is_body_pos_outside = (body_pos_env.abs() > threshold).any(dim=-1).any(dim=-1)
    assert is_body_pos_outside.shape == (env.num_envs,)
    return is_body_pos_outside

def drone_collision(
    env: ManagerBasedRLEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the drones collide."""
    robot = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = get_drone_pdist(rpos)
    separation = pdist.min(dim=-1).values.min(dim=-1).values # get the smallest distance between drones in the swarm
    is_drone_collision = (separation
        < threshold
    )
    assert is_drone_collision.shape == (env.num_envs,)
    return is_drone_collision