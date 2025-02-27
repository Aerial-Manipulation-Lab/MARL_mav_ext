# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

script_dir = Path(__file__).parent
usd_path = script_dir / "data/AMR/flycrane_rod/flycrane_rod.usd"

##
# Configuration
##

FLYCRANE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Flycrane",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "rope_1_sphere_joint_0_joint_x": 0.5,
            "rope_2_sphere_joint_0_joint_x": -0.5,
            "rope_3_sphere_joint_0_joint_y": 0.5,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Crazyflie quadcopter."""
