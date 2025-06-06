# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from MARL_mav_carry_ext.assets import FLYPENT_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Events for the hovering task.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),
            "mass_distribution_params": (1.0, 1.8),  # range of mass to randomize,
            "operation": "abs",
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (1.5, 1.5),
    #             "roll": (-0, 0),
    #             "pitch": (-0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class MARLHoverFlypentEnvCfg(DirectMARLEnvCfg):
    # control mode
    control_mode = "ACCBR"  # ACCBR or geometric
    # env
    decimation = 3
    episode_length_s = 20

    # delay parameters
    max_delay = 4  # in number of steps, with policy = 100hz -> 40ms
    constant_delay = 4  # in number of steps, with policy = 100hz -> 40ms
    # history of observations
    partial_obs = True  # if only local observations are used
    history_len = 3

    possible_agents = ["falcon1", "falcon2", "falcon3", "falcon4", "falcon5"]
    num_drones = len(possible_agents)
    if control_mode == "geometric":
        action_dim_geo = 12
        action_spaces = {"falcon1": action_dim_geo, "falcon2": action_dim_geo, "falcon3": action_dim_geo, "falcon4": action_dim_geo, "falcon5": action_dim_geo}
        obs_dim_geo = 87  # + action_dim_geo * (max_delay + 1) * num_drones # drone states, OH vector + action buffer
        observation_spaces = {"falcon1": obs_dim_geo, "falcon2": obs_dim_geo, "falcon3": obs_dim_geo, "falcon4": obs_dim_geo, "falcon5": obs_dim_geo}
        state_space = 84  # + action_dim_geo * (max_delay + 1) * num_drones # drone states, OH vector + action buffer
    elif control_mode == "ACCBR":
        action_dim_accbr = 5
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr, "falcon3": action_dim_accbr, "falcon4": action_dim_accbr, "falcon5": action_dim_accbr}
        if partial_obs:
            obs_dim_accbr = 47 * history_len
        else:
            obs_dim_accbr = (
                125  # + action_dim_accbr * (max_delay + 1) * num_drones # drone states, OH vector + action buffer
            )
        observation_spaces = {"falcon1": obs_dim_accbr, "falcon2": obs_dim_accbr, "falcon3": obs_dim_accbr, "falcon4": obs_dim_accbr, "falcon5": obs_dim_accbr}
        state_space = 120  # + action_dim_accbr * (max_delay + 1) * num_drones # drone states, OH vector + action buffer

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )
    # robot
    robot_cfg: ArticulationCfg = FLYPENT_CFG.replace(prim_path="/World/envs/env_.*/flypent")
    robot_cfg.spawn.activate_contact_sensors = True

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flypent/.*", update_period=0.0, history_length=3, debug_vis=False
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    contact_sensor_threshold = 0.1

    # falcon CoM names
    falcon_names = "Falcon.*_base_link_inertia"
    # rotor names
    falcon_rotor_names = "Falcon.*_rotor_.*"
    # payload name
    payload_name = "load_odometry_sensor_link"
    # rope name and termination terms
    bottom_rope_name = ["rope_1_link_0", "rope_2_link_0", "rope_3_link_0", "rope_4_link_0", "rope_5_link_0"]
    middle_rope_name = ["rope_1_link_1", "rope_2_link_1", "rope_3_link_1", "rope_4_link_1", "rope_5_link_1"]
    top_rope_name = ["rope_1_link_2", "rope_2_link_2", "rope_3_link_2", "rope_4_link_2", "rope_5_link_2"]
    cable_angle_limits_drone = 0.0  # cos(angle) limits
    cable_angle_limits_payload = -math.sqrt(2) / 2  # cos(angle) limits
    cable_collision_threshold = 0.2
    cable_collision_num_points = 5
    drone_collision_threshold = 0.3
    bounding_box_threshold = 5.0
    goal_achieved_range = 0.3
    goal_achieved_ori_range = 0.4
    rope_tension_threshold = 1.6 # N

    # low level control
    low_level_decimation: int = 1
    max_thrust_pp = 6.25  # N

    # rewards
    pos_track_weight = 1.5
    ori_track_weight = 1.5
    action_smoothness_weight = 0.5
    body_rate_penalty_weight = 0.5
    force_penalty_weight = 0.5
    downwash_rew_weight = 0.5

    # goal terms
    goal_range = {
        "pos_x": (-1.0, 1.0),
        "pos_y": (-1.0, 1.0),
        "pos_z": (0.5, 1.5),
        "roll": (-math.pi / 4, math.pi / 4),
        "pitch": (-math.pi / 4, math.pi / 4),
        "yaw": (-math.pi, math.pi),
    }
    range_curriculum_steps = 7500

    make_quat_unique_command = False
    # goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "goal": sim_utils.SphereCfg(
    #             radius=0.0335,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
    #         ),
    #     },
    # )

    # debug visualization
    debug_vis: bool = True
    if debug_vis:
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)

    events = EventCfg()
