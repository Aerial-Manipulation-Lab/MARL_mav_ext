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
from MARL_mav_carry_ext.splines import nonic_spline

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
    # test_times = torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]) * 2
    # test_waypoint = torch.tensor([[
    #             0.27, 0.22, 2.2,   0.0,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,
    #             0.3275, 0.05, 2.2,   0.05,  -0.02,  0.02,   -0.01, 0.01,   0.01,   0.01,  -0.005, 0.005,   0.001,  -0.002,  0.002,  # point 1
    #             0.345,  0.0,  2.22,  0.07,  -0.08,  0.05,   -0.02, 0.02,   0.02,   0.015, -0.01,  0.005,   0.002,  -0.003,  0.003,  # point 2
    #             0.375, -0.15, 2.3,   0.1,   -0.15,  0.07,   -0.03, 0.03,   0.03,   0.02,  -0.015, 0.007,   0.004,  -0.005,  0.004,  # point 3
    #             0.45,  -0.3, 2.4, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0,]])  # point 4 (unchanged)

    # # get spline coefficients
    # coeffs = nonic_spline.get_coeffs(test_waypoint, test_times, env.scene.num_envs)    
    # # evaluate spline
    # positions = []
    # velocities = []
    # accelerations = []
    # jerks = []
    # snaps = []

    # eval_times = torch.linspace(0, 2, 100)
    # for eval_time in eval_times:
    #     position, velocity, acceleration, jerk, snap = nonic_spline.evaluate_trajectory(coeffs, test_times, eval_time)
    #     positions.append(position[0].cpu())
    #     velocities.append(velocity[0].cpu())
    #     accelerations.append(acceleration[0].cpu())
    #     jerks.append(jerk[0].cpu())
    #     snaps.append(snap[0].cpu())

    # # Plot velocities against time
    # plt.figure()
    # plt.plot(eval_times, positions)
    # plt.xlabel('Time')
    # plt.ylabel('positions')
    # plt.title('positions vs Time')
    # plt.legend(['X', 'Y', 'Z'])
    # # Plot velocities against time
    # plt.figure()
    # plt.plot(eval_times, velocities)
    # plt.xlabel('Time')
    # plt.ylabel('Velocities')
    # plt.title('Velocities vs Time')
    # plt.legend(['X', 'Y', 'Z'])
    # # Plot accelerations against time
    # plt.figure()
    # plt.plot(eval_times, accelerations)
    # plt.xlabel('Time')
    # plt.ylabel('Accelerations')
    # plt.title('Accelerations vs Time')
    # plt.legend(['X', 'Y', 'Z'])
    # # Plot jerks against time
    # plt.figure()
    # plt.plot(eval_times, jerks)
    # plt.xlabel('Time')
    # plt.ylabel('Jerks')
    # plt.title('Jerks vs Time')
    # plt.legend(['X', 'Y', 'Z'])
    # # Plot snaps against time
    # plt.figure()
    # plt.plot(eval_times, snaps)
    # plt.xlabel('Time')
    # plt.ylabel('Snaps')
    # plt.title('Snaps vs Time')
    # plt.show()

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
                waypoint[:] = torch.tensor([
                # Drone 1
                [0.3275, 0.05, 2.2,   0.05,  -0.02,  0.02,   -0.01, 0.01,   0.01,  # point 1
                0.345,  0.0,  2.22,  0.07,  -0.08,  0.05,   -0.02, 0.02,   0.02, # point 2
                0.375, -0.15, 2.3,   0.1,   -0.15,  0.07,   -0.03, 0.03,   0.03,  # point 3
                0.45,  -0.3, 2.4,  # point 4 (unchanged)

                # Drone 2
                0.1,   -0.17, 2.2,  -0.03,  0.12,  -0.05,  0.02,  -0.01,  -0.01,  # point 1
                0.0,   -0.15, 2.25, -0.06,  0.1,   -0.04,  0.03,  -0.02,  -0.02,  # point 2
                -0.2,  -0.10, 2.3,  -0.1,   0.09,  -0.06,  0.04,  -0.03,  -0.03,  # point 3
                -0.3,  -0.06, 2.4,  # point 4 (unchanged)

                # Drone 3
                -0.05,  0.125, 2.2,   0.04,  0.18,  0.06,  -0.02,  0.02,   0.01,  # point 1
                0.05,   0.2,   2.25,  0.06,  0.2,   0.07,  -0.03,  0.03,   0.015,  # point 2
                0.2,    0.3,   2.35,  0.08,  0.22,  0.08,  -0.04,  0.04,   0.02,  # point 3
                0.3,    0.375, 2.4]  # point 4 (unchanged)
           ])
        #         waypoint[:] = torch.tensor([
        #         # Drone 1
        #         [0.3275, 0.05, 2.2,   0.05,  -0.02,  0.02,   -0.01, 0.01,   0.01,   0.01,  -0.005, 0.005,   0.001,  -0.002,  0.002,  # point 1
        #         0.345,  0.0,  2.22,  0.07,  -0.08,  0.05,   -0.02, 0.02,   0.02,   0.015, -0.01,  0.005,   0.002,  -0.003,  0.003,  # point 2
        #         0.375, -0.15, 2.3,   0.1,   -0.15,  0.07,   -0.03, 0.03,   0.03,   0.02,  -0.015, 0.007,   0.004,  -0.005,  0.004,  # point 3
        #         0.45,  -0.3, 2.4,  # point 4 (unchanged)

        #         # Drone 2
        #         0.1,   -0.17, 2.2,  -0.03,  0.12,  -0.05,  0.02,  -0.01,  -0.01,  0.01,  0.005,  -0.005,  -0.001, 0.002,  -0.002,  # point 1
        #         0.0,   -0.15, 2.25, -0.06,  0.1,   -0.04,  0.03,  -0.02,  -0.02,  0.015, 0.01,   -0.008,  -0.003, 0.004,  -0.003,  # point 2
        #         -0.2,  -0.10, 2.3,  -0.1,   0.09,  -0.06,  0.04,  -0.03,  -0.03,  0.02,  0.015,  -0.01,   -0.004, 0.005,  -0.004,  # point 3
        #         -0.3,  -0.06, 2.4,  # point 4 (unchanged)

        #         # Drone 3
        #         -0.05,  0.125, 2.2,   0.04,  0.18,  0.06,  -0.02,  0.02,   0.01,   0.01,  -0.01,   0.005,   0.001,  -0.002,  0.002,  # point 1
        #         0.05,   0.2,   2.25,  0.06,  0.2,   0.07,  -0.03,  0.03,   0.015,  0.015, -0.015,  0.008,   0.003,  -0.004,  0.003,  # point 2
        #         0.2,    0.3,   2.35,  0.08,  0.22,  0.08,  -0.04,  0.04,   0.02,   0.02,  -0.02,   0.01,    0.005,  -0.006,  0.004,  # point 3
        #         0.3,    0.375, 2.4]  # point 4 (unchanged)
        #    ])

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
