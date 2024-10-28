import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_inv, quat_mul


def falcon_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the falcon flies too low."""
    robot = env.scene[asset_cfg.name]
    falcon_idx = robot.find_bodies("Falcon.*base_link")[0]
    falcon_pos = robot.data.body_state_w[:, falcon_idx, :3] - env.scene.env_origins.unsqueeze(1)
    is_falcon_pos_low = (falcon_pos[..., 2] < threshold).any(dim=1)
    assert is_falcon_pos_low.shape == (env.num_envs,)
    return is_falcon_pos_low


def falcon_spin(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the falcon spins too fast."""
    robot = env.scene[asset_cfg.name]
    falcon_idx = robot.find_bodies("Falcon.*base_link")[0]
    falcon_ang_vel = robot.data.body_state_w[:, falcon_idx, 10:].reshape(env.scene.num_envs, -1)
    is_falcon_spin = (falcon_ang_vel > threshold).any(dim=1)
    assert is_falcon_spin.shape == (env.num_envs,)
    return is_falcon_spin


# def falcon_angle_sine(env: ManagerBasedRLEnv, threshold: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Terminate when the falcon angle is too large."""
#     robot = env.scene[asset_cfg.name]
#     falcon_idx = robot.find_bodies("Falcon.*base_link")[0]
#     falcon_quat = robot.data.body_state_w[:, falcon_idx, 3:7]
#     roll, pitch, yaw = euler_xyz_from_quat(falcon_quat)
#     mapped_angle = torch.stack((torch.sin(roll), torch.sin(pitch)), dim=2)
#     return (torch.abs(mapped_angle) > threshold).any(dim=2)


def payload_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload flies too low."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos = robot.data.body_state_w[:, payload_idx, :3].squeeze(1) - env.scene.env_origins
    is_payload_low = payload_pos[:, 2] < threshold
    assert is_payload_low.shape == (env.num_envs,)
    return is_payload_low


def payload_spin(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload spins too fast."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1)
    is_payload_spin = (payload_ang_vel > threshold).any(dim=1)
    assert is_payload_spin.shape == (env.num_envs,)
    return is_payload_spin


def payload_angle_cos(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload angle is too large."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    roll, pitch, yaw = euler_xyz_from_quat(payload_quat)  # yaw can be whatever
    mapped_angle = torch.stack((torch.cos(roll), torch.cos(pitch)), dim=1)
    is_angle_limit = (mapped_angle < threshold).any(dim=1)
    assert is_angle_limit.shape == (env.num_envs,)
    return is_angle_limit


def nan_states(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when any body states are NaN."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(".*")[0]
    body_states = robot.data.body_state_w[:, body_idx, :]
    is_nan_states = torch.isnan(body_states).any(dim=-1).any(dim=-1)
    assert is_nan_states.shape == (env.num_envs,)
    return is_nan_states


def large_states(
    env: ManagerBasedRLEnv, threshold: float = 1e3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when any body states are too large."""
    robot = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(".*")[0]
    body_states = robot.data.body_state_w[:, body_idx, :]
    is_large_states = (body_states.abs() > threshold).any(dim=-1).any(dim=-1)
    assert is_large_states.shape == (env.num_envs,)
    return is_large_states


def cable_angle_drones_cos(
    env: ManagerBasedRLEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle of cable between cable and drones."""
    robot = env.scene[asset_cfg.name]
    base_rope_idx = robot.find_bodies("rope_.*_link_6")[0]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    rope_orientations_world = robot.data.body_state_w[:, base_rope_idx, 3:7].view(-1, 4)
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
    base_rope_idx = robot.find_bodies("rope_.*_link_0")[0]
    payload_idx = robot.find_bodies("load_link")[0]
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
    body_idx = robot.find_bodies(".*")[0]
    body_pos = robot.data.body_state_w[:, body_idx, :3]
    body_pos_env = body_pos - env.scene.env_origins.unsqueeze(1)
    is_body_pos_outside = (body_pos_env.abs() > threshold).any(dim=-1).any(dim=-1)
    assert is_body_pos_outside.shape == (env.num_envs,)
    return is_body_pos_outside
