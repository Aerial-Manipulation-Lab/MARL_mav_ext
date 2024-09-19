import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg

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


def drone_poses(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Drone poses xyz, quat in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    drone_idx = robot.find_bodies("Falcon.*base_link")[0]
    return robot.data.body_state_w[:, drone_idx, :]  # xyz, quat, lin_vel, ang_vel


# """
# Commands
# """
# def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
#     """The generated command from command term in the command manager with the given name."""
#     return env.command_manager.get_command(command_name)
