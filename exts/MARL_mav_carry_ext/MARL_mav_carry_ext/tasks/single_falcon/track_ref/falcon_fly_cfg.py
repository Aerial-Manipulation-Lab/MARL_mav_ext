from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from MARL_mav_carry_ext.controllers import GeometricController
from MARL_mav_carry_ext.assets import FALCON_CFG
from omni.isaac.lab.markers import CUBOID_MARKER_CFG, VisualizationMarkers  # isort: skip
from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.mdp.marker_utils import ACC_MARKER_CFG, ORIENTATION_MARKER_CFG
from omni.isaac.lab.utils.math import normalize, quat_from_angle_axis, euler_xyz_from_quat

import csv

@configclass
class FalconEnvCfg(DirectRLEnvCfg):
    """Configuration for the Falcon environment."""
    episode_length_s = 100.0
    decimation = 10
    low_level_decimation = 2
    action_space = 30 # waypoint end goal
    observation_space = 19
    state_space = 0 # arbitrary for now
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        gravity=(0.0, 0.0, -9.8066),
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = FALCON_CFG.replace(prim_path="/World/Falcon/Robot")

    # rewards
    # empty for now

class FalconEnv(DirectRLEnv):
    """Environment for the single falcon."""
    cfg: FalconEnvCfg

    def __init__(self, cfg: FalconEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._rotor_idx = self._robot.find_bodies("Falcon_rotor.*")[0]
        self._falcon_idx = self._robot.find_bodies("Falcon_base_link")[0]
        self._action_space = gym.spaces.flatdim(self.single_action_space)
        self._geometric_controller = GeometricController(self.num_envs)
        self._actions = torch.zeros(self.num_envs, self._action_space, device=self.sim.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._forces = torch.zeros(self.num_envs, len(self._rotor_idx), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._low_level_decimation = self.cfg.low_level_decimation
        self._high_level_decimation = self.cfg.decimation
        self._ll_counter = 0
        self._hl_counter = 0
        self._setpoint_id = 0
        self._planner_dt = 1/self._high_level_decimation
        self._eval_time = 0.0

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.set_debug_vis(self.cfg.debug_vis)

        if self.cfg.debug_vis:
            self.des_acc_debug = torch.zeros(self.num_envs, 3, device=self.device)
            self.drone_positions_debug = torch.zeros(self.num_envs, 3, device=self.device)
            self.des_ori_debug = torch.zeros(self.num_envs, 4, device=self.device)

        # load test trajectory
        with open("/home/isaac-sim/Jack_Zeng/MARL_mav_ext/scripts/MARL_mav_carry/Jack_testing_scripts/test_trajectories/loop_10.csv", "r") as f:
            reader = csv.reader(f, delimiter=",")
            i = 0
            references = []
            for row in reader:
                if i > 0:
                    references.append([float(x) for x in row])
                i += 1
            self._references = torch.tensor(references, device=self.sim.device).repeat(self.num_envs, 1, 1)

        # with open("/home/isaac-sim/Jack_Zeng/MARL_mav_ext/scripts/MARL_mav_carry/Jack_testing_scripts/test_trajectories/circle_2m_5N.csv", "r") as f:
        #     reader = csv.reader(f, delimiter=",")
        #     i = 0
        #     references = []
        #     for row in reader:
        #         if i > 1:
        #             references.append([float(x) for x in row])
        #         i += 1
        #     self._references = torch.tensor(references, device=self.sim.device).repeat(self.num_envs, 1, 1)

    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        # apply the low level controller here
        # while (self._robot.data._sim_timestamp) > self._actions[0, 0]:
        #     # print(self._robot.data._sim_timestamp)
        #     # print(self._actions[0, 0])
        #     self._setpoint_id += 1
        #     self._actions = self._references[:, self._setpoint_id]
        #     print("sampled behind")
        # self._eval_time = (1/(self._high_level_decimation/self._low_level_decimation)) * self._planner_dt
        pass

    def _apply_action(self):
        if self._ll_counter % self._low_level_decimation == 0:
            future_setpoints = self._references[:, :, 0] > self._robot.data._sim_timestamp
            self._setpoint_id = torch.argmax(future_setpoints.float(), dim=1)
            self._actions = self._references[:, self._setpoint_id.data[0]]
            drone_states: dict = {}  # dict of tensors
            observations = self._get_observations()["policy"]
            drone_states["pos"] = observations[:, :3]
            drone_states["quat"] = observations[:, 3:7]
            drone_states["lin_vel"] = observations[:, 7:10]
            drone_states["ang_vel"] = observations[:, 10:13]
            drone_states["lin_acc"] = observations[:, 13:16]
            drone_states["ang_acc"] = observations[:, 16:]

            # linearly interpolate the setpoint
            # difference_pos = (self._actions[:, 1:4] - self._previous_actions[:, 1:4])/self._planner_dt
            # difference_vel = (self._actions[:, 8:11] - self._previous_actions[:, 8:11])/self._planner_dt
            # difference_acc = (self._actions[:, 14:17] - self._previous_actions[:, 14:17])/self._planner_dt
            # difference_jerk = (self._actions[:, 24:27] - self._previous_actions[:, 24:27])/self._planner_dt
            # difference_snap = (self._actions[:, 27:] - self._previous_actions[:, 27:])/self._planner_dt
            # difference_yaw_rate = (self._actions[:, 13] - self._previous_actions[:, 13])/self._planner_dt
            # difference_yaw_acc = (self._actions[:, 19] - self._previous_actions[:, 19])/self._planner_dt

            # self._desired_pos_w = difference_pos * self._eval_time + self._previous_actions[:, 1:4]
            self._desired_pos_w = self._actions[:, 1:4]
            # print("desired pos shape: ", self._desired_pos_w.shape)
            drone_setpoint = {}
            # drone_setpoint["pos"] = self._desired_pos_w
            # drone_setpoint["lin_vel"] = difference_vel * self._eval_time + self._previous_actions[:, 8:11]
            # drone_setpoint["lin_acc"] = difference_acc * self._eval_time + self._previous_actions[:, 14:17]
            # drone_setpoint["jerk"] = difference_jerk * self._eval_time + self._previous_actions[:, 24:27]
            # drone_setpoint["snap"] = difference_snap * self._eval_time + self._previous_actions[:, 27:]
            # _, _, yaw = euler_xyz_from_quat(actions[:, 4:8])
            # drone_setpoint["yaw"] = yaw.view(self.num_envs, 1)
            # drone_setpoint["yaw_rate"] = difference_yaw_rate * self._eval_time + self._previous_actions[:, 13]
            # drone_setpoint["yaw_acc"] = difference_yaw_acc * self._eval_time + self._previous_actions[:, 19]

            drone_setpoint["pos"] = self._desired_pos_w
            drone_setpoint["q_cmd"] = self._actions[:, 4:8]
            drone_setpoint["des_ang_vel"] = self._actions[:, 11:14]
            drone_setpoint["des_ang_acc"] = self._actions[:, 17:20]
            drone_setpoint["lin_vel"] = self._actions[:, 8:11]
            drone_setpoint["lin_acc"] = self._actions[:, 14:17]
            drone_setpoint["jerk"] = self._actions[:, 24:27]
            drone_setpoint["snap"] = self._actions[:, 27:]
            _, _, yaw = euler_xyz_from_quat(self._actions[:, 4:8])
            drone_setpoint["yaw"] = yaw.view(self.num_envs, 1)
            drone_setpoint["yaw_rate"] = self._actions[:, 13]
            drone_setpoint["yaw_acc"] = self._actions[:, 19]
            # self._desired_pos_w = torch.tensor([[0.5, -0.5, 1.0]], device=self.device)
            # drone_setpoint["pos"] = self._desired_pos_w 
            # drone_setpoint["lin_vel"] = torch.zeros(self.num_envs, 3, device=self.device)
            # drone_setpoint["lin_acc"] = torch.zeros(self.num_envs, 3, device=self.device)
            # drone_setpoint["jerk"] = torch.zeros(self.num_envs, 3, device=self.device)
            # drone_setpoint["snap"] = torch.zeros(self.num_envs, 3, device=self.device)
            # _, _, yaw = euler_xyz_from_quat(self._actions[:, 4:8])
            # drone_setpoint["yaw"] = torch.tensor([[0.0]], device=self.device)
            # drone_setpoint["yaw_rate"] = torch.tensor([[0.0]], device=self.device)
            # drone_setpoint["yaw_acc"] = torch.tensor([[0.0]], device=self.device)

            drone_thrusts, acc_cmd, q_cmd, torques = self._geometric_controller.getCommand(
                        drone_states, self._forces, drone_setpoint
                    )
            if self.cfg.debug_vis:
                self.des_acc_debug = acc_cmd
                self.drone_positions_debug = drone_states["pos"]
                self.des_ori_debug = q_cmd

            # self._eval_time += (1/(self._high_level_decimation/self._low_level_decimation)) * self._planner_dt
            self._forces[..., 2] = drone_thrusts
            self._moments[..., :] = torques
            self._ll_counter = 0

        self._ll_counter += 1
        # self._hl_counter += 1
        self._robot.set_external_force_and_torque(self._forces, torch.zeros_like(self._forces), body_ids=self._rotor_idx)
        self._robot.set_external_force_and_torque(torch.zeros_like(self._moments), self._moments, body_ids=self._falcon_idx)

    def _get_observations(self) -> dict:
        # observations from the example, not real ones
        drone_idx = self._robot.find_bodies("Falcon_base_link")[0]
        obs = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_w,
                self._robot.data.body_lin_acc_w[:, drone_idx].squeeze(1),
                self._robot.data.body_ang_acc_w[:, drone_idx].squeeze(1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # rewards from the example, not real ones
        rewards = torch.zeros(self.num_envs, device=self.device)
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0)
        return died, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._robot.set_external_force_and_torque(torch.zeros(self.num_envs, 3, 3, device=self.device), torch.zeros_like(self._forces))

    def _set_debug_vis_impl(self, debug_vis: bool):
    # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            if not hasattr(self, "acc_marker"):
                marker_cfg = ACC_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_acc"
                self.acc_marker = VisualizationMarkers(marker_cfg)
            self.acc_marker.set_visibility(False)
            if not hasattr(self, "drone_ori_marker"):
                marker_cfg = ORIENTATION_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_ori"
                self.drone_ori_marker = VisualizationMarkers(marker_cfg)
            self.drone_ori_marker.set_visibility(False)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "acc_marker"):
                self.acc_marker.set_visibility(False)
            if hasattr(self, "drone_ori_marker"):
                self.drone_ori_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # drone desired accelerations

        # Normalize the desired direction vector (which represents the direction)
        acc_orientation_axis = normalize(self.des_acc_debug)
        # Define the default x-axis (the direction the arrow marker points to by default)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # Compute the dot product between x-axis and desired direction to check alignment
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        # Flip the desired direction if the dot product is negative (indicating opposite direction)
        mask = (cos_angle.view(-1, 1) < 0).squeeze()
        # acc_orientation_axis = torch.where(mask.squeeze(), -acc_orientation_axis.view(-1,3), acc_orientation_axis.view(-1,3))
        acc_orientation_axis.view(-1, 3)[mask] = -acc_orientation_axis.view(-1, 3)[mask]
        # Compute the axis of rotation (cross product between x-axis and desired direction)
        rotation_axis = torch.linalg.cross(x_axis, acc_orientation_axis)
        # Compute the angle between x-axis and desired direction using dot product
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # Clamp to avoid numerical issues
        # Handle cases where the vectors are parallel (no rotation needed)
        rotation_axis = torch.where(
            torch.norm(rotation_axis, dim=-1, keepdim=True) < 1e-6,  # Check if parallel
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # Default to any orthogonal axis
            normalize(rotation_axis),
        )
        # Compute the quaternion from the angle-axis representation
        acc_orientation = quat_from_angle_axis(angle.view(-1), rotation_axis.view(-1, 3)).view(-1, 4)

        # Visualize the arrow marker with the new orientation
        self.acc_marker.visualize(
            self.drone_positions_debug.view(-1, 3),
            acc_orientation.view(-1, 4),
            self.des_acc_debug.view(-1, 3) / 5,
            marker_indices=[0] * self.num_envs,
        )

        self.drone_ori_marker.visualize(
            self.drone_positions_debug.view(-1, 3),
            self.des_ori_debug.view(-1, 4),
            marker_indices=[0] * self.num_envs,
        )