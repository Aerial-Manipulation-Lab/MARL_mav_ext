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

from isaaclab.app import AppLauncher

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

import csv
import gymnasium as gym
import matplotlib.pyplot as plt

from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import MARLHoverEnvCfg
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env import MARLHoverEnv

from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.dict import print_dict


def main():
    """Main function."""
    # create environment config
    env_cfg = MARLHoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = MARLHoverEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./marl_videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    count = 0

    # references for testing

    stretch_position = torch.tensor(
        [
            [
                0.27,
                1.0867,
                1.7,
                0.27,
                -1.0867,
                1.7,
                -1.1367,
                0.0,
                1.7,
            ]
        ],
        dtype=torch.float32,
    )

    falcon1_geo_tensor = torch.zeros((env.num_envs, 12), device= env.device)
    falcon2_geo_tensor = torch.zeros((env.num_envs, 12), device= env.device)
    falcon3_geo_tensor = torch.zeros((env.num_envs, 12), device= env.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            # step the environment
            if count % 500 == 0:
                env.reset()
            falcon1_geo_tensor[:, 0:3] = stretch_position[:, 0:3]
            falcon2_geo_tensor[:, 0:3] = stretch_position[:, 3:6]
            falcon3_geo_tensor[:, 0:3] = stretch_position[:, 6:9]
            action = {"falcon1": falcon1_geo_tensor,
                      "falcon2": falcon2_geo_tensor,
                      "falcon3": falcon3_geo_tensor,}
            obs, rew, terminated, truncated, info = env.step(action)
            terminated = list(terminated.values())[0]
            truncated = list(truncated.values())[0]
            if any(terminated) or any(truncated):
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # update counter

            # if args_cli.video:
            #     if count/2 == args_cli.video_length:
            # break
            count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
