# Multi-Agent reinforcement learning for multi-drone transport system

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

Author - [Jack Zeng](https://github.com/Jackkert)

## Overview
This repository is an NVIDIA Isaac Lab extension that contains the environment and algorithm to control a multi-drone transport system (flycrane). As of now, 1 task has been implemented:

- `Hover` The flycrane makes the payload hover at the intial pose:


https://github.com/user-attachments/assets/3da98247-0004-44f3-95c0-acbaa4afa166

In the video, the flycrane during training can be seen holding the payload in place.

*Tasks in progress*:

- Hover the payload at different reference positions

*Tasks to be implemented*:

- `Track` Make the payload follow a given trajectory
- `FlyThrough` Make the payload fly through a gap/between 2 walls

### Installation

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Using a python interpreter that has Isaac Lab installed, install the library

```
cd exts/MARL_mav_carry_ext
python -m pip install -e .
```

## Assets

The assets used for the tasks are under [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets).
The assets folder contains `$(ROBOT).py` files which have the configuration the respective robot in the form an `ArticulationCfg` (Isaac Lab actuated robot config class).

Then, in the [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR) folder, the robot URDF and corresponding meshes are located in the `$(ROBOT)_data` folder and the USD files can be found in the `$(ROBOT)` folder.

## Environments

As of now, only the hover environment has been implemented. The environments can be found under [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry).

### Flycrane hover environment

In the [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover) folder, the environment can be found in `hover_env_cfg.py`. Following the Isaac Lab structure the environment is implemented as a [`ManagerBasedRLEnv`](https://isaac-sim.github.io/IsaacLab/source/api/lab/omni.isaac.lab.envs.html#omni.isaac.lab.envs.ManagerBasedRLEnv). Later, the environment will likely be rewritten to the [`DirectMARLEnv`](https://isaac-sim.github.io/IsaacLab/source/api/lab/omni.isaac.lab.envs.html#omni.isaac.lab.envs.DirectMARLEnv) class.

The manager based environment consists of multiple modules, and their configs can be found in the `hover_env_cfg.py` file:

- `Scene` The scene that describes the environment. This describes all prims present in the environment such as ground plane, lights, robots and sensors.
- `CommandManager` Generate a new pose command as a reference for the payload and resample at certain time intervals.
- `ActionManager` Processes and sends the actions to the simulation. Now, this class only has 1 `ActionTerm` that purely reshapes and clamps the forces and torques applied to the body. The RL policy directly learns the forces and torques on the drone bodies (collective thrust and 3 torques). Later, the plan is to add a low-level controller here.
- `ObservationManager` The observations available to the policy. Currently, the problem is handled as a centralized problem and contains the following observations:
    - Payload pose (positions, orientations, linear velocities and angular velocities) in environment frame
    - Drone poses (positions, orientations, linear velocities and angular velocities) in environment frame
    - Payload positions and orientation errors to the goal
    - Relative position of drone to the payload
    - Drone positions relative to eachother
    - Euclideans distance between the drones
- `EventManager` Describes what happens on certain events such as startup, reset or at certain time intervals. Right now, when reset is called, velocities and external forces/torques on all bodies are set to 0. The pose of the flycrane is sampled from a uniform distribution to randomize the initial state.
- `RewardManager` Implements the reward function. Consisting of the following terms:
    - `reward_separation` Reward for keeping the drones at a safe distance from each other
    - `reward_pose` Reward for tracking the payload pose
    - `reward_up` Reward for keeping the payload upright
    - `reward_spin_payload` Reward for keeping angular velocities of the payload small
    - `reward_swing` Reward for keeping linear velocities of the payload small
    - `reward_effort` Reward to keep the effort small
    - `reward_spin_drones` Reward to keep the drone angular velocities small


```
reward = reward_separation * (
        reward_pose
        + reward_pose * (reward_up + reward_spin_payload + reward_swing)
        + reward_effort
        + reward_spin_drones
    )
```

- `TerminationsManager` Terminates the episode corresponding to an environment if a termination condition has been met. The implemented termination terms are"
    - `time_out` Time out after max episode length
    - `falcon_fly_low` Terminate when drones fly too low
    - `payload_fly_low` Terminate when the payload flies too low
    - `illegal_contact` Terminate when forces between bodies get too large
    - `payload_spin` Terminate when the angular velocities of the payload gets too large
    - `payload_angle`Terminate when the linear velocities of the payload gets too large
    - `falcon_spin`Terminate when the angular velocities of the drones gets too large

- `CurriculumManager` Manager of different learning tasks in order to increase difficulty of the task over time (curriculum learning). This is not implemented.

All helper functions are implemented in the [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover/mdp](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover/mdp) folder in their respective python file.

## Training and playing
Isaac Lab offers different wrappers for different RL libraries to make it easy to switch between libraries. The scripts for the corresponding libraries are implemented in [scripts](https://github.com/Jackkert/MARL_mav_ext/tree/main/scripts). The useable libraries are rl_games, rsl_rl and skrl.

### Agents
The agent configurations for the flycrane are under [exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover/config/flycrane/agents](https://github.com/Jackkert/MARL_mav_ext/tree/main/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/MARL_mav_carry/hover/config/flycrane/agents). The environments are registered as a gym environment and the parameters of the agents can be changed here.

### Training
To train the agent, for example using rsl_rl. You can run the following command from the command line:

`python3 scripts/rsl_rl/train.py --task=Isaac-flycrane-payload-hovering-v0 --num_envs=1024 --video --video_interval=20000 --headless`

This will start the training for the hover task with the configured settings in the agent configuration file. For more command line interface arguments, check the respective `train.py` file under `scripts/`.


### Playing
To play with the learned agent, you can run the `play.py` script. This will load the latest checkpoint from the `logs` that have been accumulated during training. For this, execute (for example):

`python3 scripts/rsl_rl/play.py --task=Isaac-flycrane-payload-hovering-v0 --num_envs=1`

To enable debug visualizations of the executed forces and torques, as well as goal poses, enable the debug_vis parameter. (more on this soon)
To gather data and plot results of the played episode, add the `--plot_data=True` flag, this will plot several statistics against time: (coming soon)

- Payload pose: positions, orientations, linear velocities, angular velocities
- Drone poses: positions, orientations, linear velocities, angular velocities
- Drone actions: forces and torques on each on each drone body
- Cable angles (orientation of the link attached to the payload)
- Payload positional error to goal
- Payload orientation error to goal

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
