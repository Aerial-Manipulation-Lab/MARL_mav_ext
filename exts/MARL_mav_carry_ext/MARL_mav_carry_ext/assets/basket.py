# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

script_dir = Path(__file__).parent
usd_path = script_dir / "data/AMR/basket/basket.usd"

##
# Configuration
##

BASKET_CFG = RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/payload",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path.resolve()),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )