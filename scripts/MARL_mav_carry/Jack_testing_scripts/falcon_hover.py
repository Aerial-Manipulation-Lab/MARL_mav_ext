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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
from MARL_mav_carry_ext.assets import FALCON_CFG  # isort:skip


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=0.005))
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = FALCON_CFG
    robot_cfg.spawn.func("/World/Falcon/Robot_1", robot_cfg.spawn, translation=(1.5, 0.5, 0.42))
    robot_cfg.spawn.func("/World/Falcon/Robot_2", robot_cfg.spawn, translation=(-1.5, 0.5, 0.42))

    # create handles for the robots
    robot = Articulation(robot_cfg.replace(prim_path="/World/Falcon/Robot.*"))

    # Play the simulator
    sim.reset()

    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("Falcon_base_link")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    print("falcon mass: ", robot_mass)
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot_root_state = robot.data.default_root_state[:, :7]
    robot_root_state[0, :3] += torch.tensor([1.5, 0.5, 0.42])
    robot_root_state[0, 3:7] = torch.tensor([0.4645017, 0.1911519, 0.4645017, 0.7293403])
    robot_root_state[1, :3] += torch.tensor([-1.5, 0.5, 0.42])
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot_root_state)
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        robot_state = robot.data.body_state_w[0, prop_body_ids, :]
        print(f"robot position: {robot_state[0, :3]}")
        print(f"robot orientation: {robot_state[0, 3:7]}")
        print(f"robot linear velocity: {robot_state[0, 7:10]}")
        print(f"robot angular velocity: {robot_state[0, 10:]}")
        # apply action to the robot (make the robot float in place)
        forces = torch.zeros(robot.num_instances, len(prop_body_ids), 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = robot_mass * gravity / (len(prop_body_ids) * robot.num_instances)
        torques[..., 2] = 1
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
