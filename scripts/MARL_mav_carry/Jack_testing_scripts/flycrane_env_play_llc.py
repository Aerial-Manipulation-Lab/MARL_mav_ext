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
from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg import HoverEnvCfg_llc
from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.mdp.utils import quintic_trajectory_3d, minimum_snap_spline_3d, compute_derivatives_3d, evaluate_trajectory_3d

from omni.isaac.lab.envs import ManagerBasedRLEnv


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
    waypoints_3d = torch.tensor([
        [0, 0, 0],
        [5, 5, 5],
        [10, 0, 10],
        [15, -5, 5]
    ], dtype=torch.float32)  # Positions (x, y, z) at each time point
    times = torch.tensor([0, 2, 4, 6], dtype=torch.float32)  # Timestamps

    # Generate the minimum snap spline
    coeffs_list_3d = minimum_snap_spline_3d(waypoints_3d, times)

    # Evaluate the trajectory at different time points
    t_eval = 3.0  # Evaluate at t = 3 seconds
    position_at_t, velocity_at_t, acceleration_at_t, jerk_at_t, snap_at_t = evaluate_trajectory_3d(coeffs_list_3d, times, t_eval)

    print(f"Position at t={t_eval}: {position_at_t}")
    print(f"Velocity at t={t_eval}: {velocity_at_t}")
    print(f"Acceleration at t={t_eval}: {acceleration_at_t}")
    print(f"Jerk at t={t_eval}: {jerk_at_t}")
    print(f"Snap at t={t_eval}: {snap_at_t}")

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
