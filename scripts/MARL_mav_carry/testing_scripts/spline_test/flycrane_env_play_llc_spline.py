# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during execution.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import matplotlib.pyplot as plt

from MARL_mav_carry_ext.splines import septic_spline
from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg_spline import HoverEnvCfg_llc_spline

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.dict import print_dict

# test_times = torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]) * 2
# starting_point = torch.tensor([0.27, 0.22, 2.1410, 0, 0, 0, 0, 0, 0])

# # get spline coefficients
# coeffs = septic_spline.get_coeffs(test_waypoint, test_times, env.scene.num_envs)
# # evaluate spline
# positions = []
# velocities = []
# accelerations = []
# jerks = []
# snaps = []

# eval_times = torch.linspace(0, 2, 100)
# for eval_time in eval_times:
#     position, velocity, acceleration, jerk, snap = septic_spline.evaluate_trajectory(coeffs, test_times, eval_time)
#     positions.append(position[0].cpu())
#     velocities.append(velocity[0].cpu())
#     accelerations.append(acceleration[0].cpu())
#     jerks.append(jerk[0].cpu())
#     snaps.append(snap[0].cpu())

#     # Plot velocities against time
#     plt.figure()
#     plt.plot(eval_times, positions)
#     plt.xlabel('Time')
#     plt.ylabel('positions')
#     plt.title('positions vs Time')
#     plt.legend(['X', 'Y', 'Z'])
#     # Plot velocities against time
#     plt.figure()
#     plt.plot(eval_times, velocities)
#     plt.xlabel('Time')
#     plt.ylabel('Velocities')
#     plt.title('Velocities vs Time')
#     plt.legend(['X', 'Y', 'Z'])
#     # Plot accelerations against time
#     plt.figure()
#     plt.plot(eval_times, accelerations)
#     plt.xlabel('Time')
#     plt.ylabel('Accelerations')
#     plt.title('Accelerations vs Time')
#     plt.legend(['X', 'Y', 'Z'])
#     # Plot jerks against time
#     plt.figure()
#     plt.plot(eval_times, jerks)
#     plt.xlabel('Time')
#     plt.ylabel('Jerks')
#     plt.title('Jerks vs Time')
#     plt.legend(['X', 'Y', 'Z'])
#     # Plot snaps against time
#     plt.figure()
#     plt.plot(eval_times, snaps)
#     plt.xlabel('Time')
#     plt.ylabel('Snaps')
#     plt.title('Snaps vs Time')
#     plt.show()


def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc_spline()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                waypoint = torch.zeros_like(env.action_manager.action)
                waypoint[:] = torch.tensor(
                    [
                        [
                            0.5,
                            -0.5,
                            2.5,  # end goal drone 1
                            -0.5,
                            0.0,
                            2.5,  # end goal drone 2
                            0.5,
                            0.5,
                            2.5,  # end goal drone 3]
                        ]
                    ],
                    dtype=torch.float32,
                )
                # waypoint[1] = torch.tensor([[0.5, 0.5, 2.5, # end goal drone 1
                #                             0.5, 0.0, 2.5, # end goal drone 2
                #                             -0.5, 0.5, 3.0, # end goal drone 3]
                #                             ]], dtype=torch.float32)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            # update counter
            count += 1

            if args_cli.video:
                if count == args_cli.video_length:
                    break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
