import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_inv, quat_mul

from .utils import get_drone_pdist, get_drone_rpos

# Body indices found in the scene
# payload_idx = [0]
# drone_idx = [71, 72, 73]
# base_rope_idx = [8, 9, 10]
# top_rope_idx = [62, 63, 64]

# for the case when the rod is used
payload_idx = [0]
drone_idx = [17, 18, 19]
base_rope_idx = [8, 9, 10]
top_rope_idx = [8, 9, 10]


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

def nan_obs(
    env: ManagerBasedRLEnv, group_name: str) -> torch.Tensor:
    """Terminate when any observation is NaN."""
    obs = env.obs_buf
    if obs: # prevent error when obs is empty dict
        is_nan_obs = torch.isnan(obs[group_name]).any(dim=-1)
        assert is_nan_obs.shape == (env.num_envs,)
        return is_nan_obs
    else:
        is_nan_obs = torch.tensor([False]*env.num_envs, device=env.sim.device)
        assert is_nan_obs.shape == (env.num_envs,)
        return is_nan_obs

def drone_collision(
    env: ManagerBasedRLEnv, threshold: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the drones collide."""
    robot = env.scene[asset_cfg.name]
    drone_pos_world_frame = robot.data.body_state_w[:, drone_idx, :3]
    rpos = get_drone_rpos(drone_pos_world_frame)
    pdist = get_drone_pdist(rpos)
    separation = pdist.min(dim=-1).values.min(dim=-1).values  # get the smallest distance between drones in the swarm
    is_drone_collision = separation < threshold
    assert is_drone_collision.shape == (env.num_envs,)
    return is_drone_collision

def cable_collision(
    env: ManagerBasedRLEnv, threshold: float = 0.0, num_points: int = 5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Check for collisions between cables.

    A collision is detected if the minimum Euclidean distance between any two points
    on different cables is below the threshold.
    """
    robot = env.scene[asset_cfg.name]
    cable_bottom_pos_env = robot.data.body_state_w[:, base_rope_idx, :3] - env.scene.env_origins.unsqueeze(1)
    cable_top_pos_env = robot.data.body_state_w[:, drone_idx, :3] - env.scene.env_origins.unsqueeze(1)
    cable_directions = cable_top_pos_env - cable_bottom_pos_env  # (num_envs, num_cables, 3)

    # Create linearly spaced points for interpolation (num_points,)
    linspace_points = torch.linspace(0, 1, num_points, device=env.device).view(1, 1, num_points, 1)  # (1, 1, num_points, 1)

    # Compute cable points (num_envs, num_cables, num_points, 3)
    cable_points = cable_bottom_pos_env.unsqueeze(2) + linspace_points * cable_directions.unsqueeze(2)  # (num_envs, num_cables, num_points, 3)

    # Flatten cable points for easier distance calculation (num_envs, num_cables * num_points, 3)
    cable_points_flat = cable_points.view(env.num_envs, -1, 3)

    # Pairwise distance calculation
    cable_points_a = cable_points_flat.unsqueeze(2)  # (num_envs, num_points_total, 1, 3)
    cable_points_b = cable_points_flat.unsqueeze(1)  # (num_envs, 1, num_points_total, 3)
    pairwise_diff = cable_points_a - cable_points_b  # (num_envs, num_points_total, num_points_total, 3)
    pairwise_distances = torch.norm(pairwise_diff, dim=-1)  # (num_envs, num_points_total, num_points_total)

    # Mask to ignore self-distances and distances within the same cable
    num_cables = cable_bottom_pos_env.shape[1]
    points_per_cable = num_points

    # Create mask to ignore points on the same cable
    cable_indices = torch.arange(num_cables, device=env.device).repeat_interleave(points_per_cable)  # (num_points_total,)
    same_cable_mask = cable_indices.unsqueeze(0) == cable_indices.unsqueeze(1)  # (num_points_total, num_points_total)
    same_cable_mask = same_cable_mask.unsqueeze(0).expand(env.num_envs, -1, -1)  # (num_envs, num_points_total, num_points_total)

    # Apply mask: set ignored distances to a large value
    pairwise_distances[same_cable_mask] = 1000.0

    # Find the minimum distance across all points in each environment
    min_distances, _ = torch.min(pairwise_distances.view(env.num_envs, -1), dim=-1)  # Shape: (num_envs,)

    # Check if the minimum distance is below the threshold
    is_cable_collision = min_distances < threshold  # Shape: (num_envs,)

    assert is_cable_collision.shape == (env.num_envs,)
    return is_cable_collision


def payload_target_distance(
        env: ManagerBasedRLEnv, threshold: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the payload is outisde a certain distance of the target."""
    robot = env.scene[asset_cfg.name]
    payload_pos_world = robot.data.body_state_w[:, payload_idx, :3].squeeze(1)
    payload_pos_env = payload_pos_world - env.scene.env_origins

    desired_pos = env.command_manager.get_command(command_name)[
        ..., :3
    ] 

    # for the trajectory case
    if len(desired_pos.shape) > 2:
        desired_pos = desired_pos[:,0]

    target_rpos = desired_pos - payload_pos_env
    dist = target_rpos.norm(dim=-1)
    is_target_far = dist > threshold

    assert is_target_far.shape == (env.num_envs,)
    return is_target_far

def sim_time_exceed(
    env: ManagerBasedRLEnv, command_name: str = "pose_twist_command"
) -> torch.Tensor:
    """Terminate when the simulation time exceeds the threshold (end of reference trajectory)."""
    command_term = env.command_manager._terms[command_name]
    is_sim_time_exceeded = command_term.sim_time > command_term.reference[0, -1, 0]

    assert is_sim_time_exceeded.shape == (env.num_envs,)
    return is_sim_time_exceeded