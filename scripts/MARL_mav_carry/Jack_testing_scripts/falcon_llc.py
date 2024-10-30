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
import matplotlib.pyplot as plt
import gymnasium as gym

from MARL_mav_carry_ext.tasks.single_falcon.track_ref import FalconEnvCfg, FalconEnv

from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.utils.dict import print_dict

def main():
    """Main function."""
    # create environment config
    env_cfg = FalconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = FalconEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    count = 0

    # test trajectory
    with open("/home/isaac-sim/Jack_Zeng/MARL_mav_ext/scripts/MARL_mav_carry/Jack_testing_scripts/test_trajectories/loop_10.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        i = 0
        references = []
        for row in reader:
            if i > 0:
                references.append([float(x) for x in row])
            i += 1
        references = torch.tensor(references, device=env.sim.device).repeat(env.num_envs, 1, 1)
            
    while simulation_app.is_running():
        with torch.inference_mode():
            count = count % references.shape[1]
            waypoint = references[:, count]
            # step the environment
            obs, rew, terminated, truncated, info = env.step(waypoint)
            # update counter
            count += 2
            
            if args_cli.video:
                if count/2 == args_cli.video_length:
                    break

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
