import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat


def falcon_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the falcon flies too low."""
    robot = env.scene[asset_cfg.name]
    falcon_idx = robot.find_bodies("Falcon.*base_link")[0]
    falcon_pos = robot.data.body_state_w[:, falcon_idx, :3]
    return (falcon_pos[..., 2] < threshold).any(dim=1)


def payload_fly_low(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload flies too low."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    return payload_pos[:, 2] < threshold


def payload_spin(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload spins too fast."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_ang_vel = robot.data.body_state_w[:, payload_idx, 10:].squeeze(1)
    return (payload_ang_vel > threshold).any(dim=1)


def payload_angle_sine(
    env: ManagerBasedRLEnv, threshold: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload angle is too large."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_quat = robot.data.body_state_w[:, payload_idx, 3:7].squeeze(1)
    roll, pitch, yaw = euler_xyz_from_quat(payload_quat) # yaw can be whatever
    mapped_angle = torch.stack((torch.sin(roll), torch.sin(pitch)), dim=1)
    return (torch.abs(mapped_angle) > threshold).any(dim=1)
