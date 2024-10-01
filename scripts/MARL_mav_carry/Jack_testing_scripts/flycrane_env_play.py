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

from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover.hover_env_cfg import HoverEnvCfg

from omni.isaac.lab.envs import ManagerBasedRLEnv


def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot_mass = env.scene["robot"].root_physx_view.get_masses().sum()
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.sim.device).norm()
    falcon_mass = 0.6 + 0.0042 * 4 + 0.00002
    rope_mass = 0.0033692587500000004 * 7 + 0.001 * 14
    payload_mass = 1.4 + 0.00001 + 0.006
    mass_left_side = 2 * falcon_mass + 2 * rope_mass + 0.5 * payload_mass
    mass_right_side = falcon_mass + rope_mass + 0.5 * payload_mass
    # print(env.scene["robot"].root_physx_view.get_masses())
    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            waypoint = torch.zeros_like(env.action_manager.action)
            waypoint[:, 0] = mass_left_side * gravity / 2
            waypoint[:, 4] = mass_left_side * gravity / 2
            waypoint[:, 8] = mass_right_side * gravity
            # waypoint[:, 6] = 0.05

            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint * 1)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
