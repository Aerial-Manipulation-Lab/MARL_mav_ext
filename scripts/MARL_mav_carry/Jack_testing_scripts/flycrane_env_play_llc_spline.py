# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from gymnasium.spaces import Box

from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg_spline import HoverEnvCfg_llc_spline
from MARL_mav_carry_ext.splines import septic_spline

from omni.isaac.lab.envs import ManagerBasedRLEnv
import matplotlib.pyplot as plt
import math

def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc_spline()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.action_space = Box(-1.0, 1.0, shape=(env.scene.num_envs, 12), dtype="float32")
    robot_mass = env.scene["robot"].root_physx_view.get_masses().sum()
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.sim.device).norm()
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14
    payload_mass = 1.4 + 0.00001 + 0.006
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                waypoint = torch.zeros_like(env.action_manager.action)
                waypoint[:] = torch.tensor([[0.5, -0.5, 2.5, # end goal drone 1
                                            -0.5, 0.0, 2.5, # end goal drone 2
                                            0.5, 0.5, 2.5, # end goal drone 3]
                                            ]], dtype=torch.float32)
                waypoint[1] = torch.tensor([[0.5, 0.5, 2.5, # end goal drone 1
                                            0.5, 0.0, 2.5, # end goal drone 2
                                            -0.5, 0.5, 3.0, # end goal drone 3]
                                            ]], dtype=torch.float32)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            # update counter
            count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
