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

from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg import HoverEnvCfg_llc
from MARL_mav_carry_ext.splines import minimum_snap_spline, evaluate_trajectory

from omni.isaac.lab.envs import ManagerBasedRLEnv
import matplotlib.pyplot as plt
import math

def main():
    """Main function."""
    # create environment config
    env_cfg = HoverEnvCfg_llc()
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
    # print(env.scene["robot"].root_physx_view.get_masses())
    # simulate physics
    # Example usage: define 3D waypoints and corresponding timestamps
    waypoints_3d = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,# point 1
                                 5, 5, 5, 1, -1, 1, 1, -5, 1, 0.0007963, 0, 0, 1, # point 2
                                 10, 0, 10, 1, -1, -1, 1, -1, -1, 0.4999998, -0.4996018, 0.4999998, 0.5003982, # point 3
                                 15, -5, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32) # point 4 # Positions (x, y, z) at each time point
    times = torch.tensor([0, 2, 4, 6], dtype=torch.float32)  # Timestamps

    # Generate the minimum snap spline
    coeffs_list_3d, orientations_traj = minimum_snap_spline(waypoints_3d, times)

    # Evaluate the trajectory at different time points
    positions = []
    velocities = []
    accelerations = []
    orientations = []
    angular_rates = []
    eval_times = torch.linspace(0, 6, 100)
    for t_eval in eval_times:
        position, velocity, acceleration, jerk, snap, orientation, angular_velocity = evaluate_trajectory(coeffs_list_3d, orientations_traj, times, t_eval)
        positions.append(position)
        velocities.append(velocity)
        accelerations.append(acceleration)
        orientations.append(orientation)
        angular_rates.append(angular_velocity)

    positions = torch.stack(positions, dim=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot velocities against time
    plt.figure()
    plt.plot(eval_times, velocities)
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.title('Velocities vs Time')
    plt.legend(['X', 'Y', 'Z'])

    # Plot accelerations against time
    plt.figure()
    plt.plot(eval_times, accelerations)
    plt.xlabel('Time')
    plt.ylabel('Accelerations')
    plt.title('Accelerations vs Time')
    plt.legend(['X', 'Y', 'Z'])

    # plot orientation against time
    orientations = torch.stack(orientations, dim=0)
    plt.figure()
    plt.plot(eval_times, orientations)
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Orientation vs Time')
    plt.legend(['W','X', 'Y', 'Z'])

    # plot angular rates against time
    angular_rates = torch.stack(angular_rates, dim=0)
    plt.figure()
    plt.plot(eval_times, angular_rates) 
    plt.xlabel('Time')
    plt.ylabel('Angular Rates')
    plt.title('Angular Rates vs Time')
    plt.legend(['X', 'Y', 'Z'])

    plt.show()

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 500 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # print("obs manager", env.observation_manager.compute())
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
