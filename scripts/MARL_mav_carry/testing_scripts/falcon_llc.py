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

from MARL_mav_carry_ext.tasks.single_falcon.track_ref import FalconEnv, FalconEnvCfg

from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.timer import Timer


def main():
    """Main function."""
    # create environment config
    env_cfg = FalconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = FalconEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": "./falcon_videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # step the environment

            obs, rew, terminated, truncated, info = env.step(torch.tensor([0.0], device=env.device))
            if terminated.any() | truncated.any():
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # update counter

            # if args_cli.video:
            #     if count/2 == args_cli.video_length:
            # break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
