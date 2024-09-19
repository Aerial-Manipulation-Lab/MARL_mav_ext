from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg

def falcon_fly_low(env: ManagerBasedRLEnv, threshold: float = 0.1, 
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> bool:
    """Terminate when the falcon flies too low."""
    robot = env.scene[asset_cfg.name]
    falcon_idx = robot.find_bodies("Falcon.*base_link")[0]
    falcon_pos = robot.data.body_state_w[:, falcon_idx, :3]
    return (falcon_pos[..., 2] < threshold).any()

def payload_fly_low(env: ManagerBasedRLEnv, threshold: float = 0.1, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> bool:
    """Terminate when the payload flies too low."""
    robot = env.scene[asset_cfg.name]
    payload_idx = robot.find_bodies("load_link")[0]
    payload_pos = robot.data.body_state_w[:, payload_idx, :3]
    return (payload_pos[..., 2] < threshold).any()