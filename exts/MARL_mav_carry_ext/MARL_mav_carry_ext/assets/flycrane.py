# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

script_dir = Path(__file__).parent
usd_path = script_dir / "data/AMR/flycrane_offset/flycrane.usd"

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
            ".*": 0.0,
        },
        joint_vel={
            "Falcon1_rotor_0_joint": 0.0,
            "Falcon1_rotor_1_joint": -0.0,
            "Falcon1_rotor_2_joint": 0.0,
            "Falcon1_rotor_3_joint": -0.0,
            "Falcon2_rotor_0_joint": 0.0,
            "Falcon2_rotor_1_joint": -0.0,
            "Falcon2_rotor_2_joint": 0.0,
            "Falcon2_rotor_3_joint": -0.0,
            "Falcon3_rotor_0_joint": 0.0,
            "Falcon3_rotor_1_joint": -0.0,
            "Falcon3_rotor_2_joint": 0.0,
            "Falcon3_rotor_3_joint": -0.0,
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
