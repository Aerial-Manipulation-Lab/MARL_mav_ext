from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import yaw_quat, euler_xyz_from_quat, quat_inv, quat_mul, quat_from_angle_axis, normalize

from .marker_utils import FORCE_MARKER_Z_CFG, GOAL_POS_MARKER_CFG, ACC_MARKER_CFG, ORIENTATION_MARKER_CFG, BLUE_ARROW_MARKER_CFG
from .observations import *
from MARL_mav_carry_ext.splines import minimum_snap_spline, evaluate_trajectory
from MARL_mav_carry_ext.controllers import GeometricController

class LowLevelAction(ActionTerm):
    """Low level action term for the hover task."""

    cfg: LowLevelActionCfg

    def __init__(self, cfg: LowLevelActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_ids = self._robot.find_bodies(cfg.body_name)[0]
        self._forces = torch.zeros(env.scene.num_envs, len(self._body_ids), 3, device=self.device)
        self._torques = torch.zeros(env.scene.num_envs, len(self._body_ids), 3, device=self.device)
        self._num_drones = cfg.num_drones
        self._waypoint_dim = cfg.waypoint_dim # pos, vel, acc, att
        self._num_waypoints = cfg.num_waypoints
        self._times = (torch.arange(self._num_waypoints + 1).float())/self._num_waypoints # normalized time vector
        self._time_horizon = cfg.time_horizon # sec
        self._waypoints = torch.zeros(env.scene.num_envs, self._num_drones * self._waypoint_dim * self._num_waypoints, device=self.device)
        self._geometric_controller = GeometricController()
        self.cfg = cfg
        self._counter = 0

        # debug
        if cfg.debug_vis:
            self.drone_positions_debug = torch.zeros(self._num_drones, 3, device=self.device)
            self.drone_goals_debug = torch.zeros(self._num_drones, 3, device=self.device)
            self.des_acc_debug = torch.zeros(self._num_drones, 3, device=self.device)
            self.des_ori_debug = torch.zeros(self._num_drones, 4, device=self.device)
            self.z_b_debug = torch.zeros(self._num_drones, 3, device=self.device)

        """
        properties
        """

    @property
    def action_dim(self) -> int:
        return self._num_drones * self._waypoint_dim * self._num_waypoints
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._waypoints

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._forces

    def process_actions(self, waypoints: torch.Tensor):
        """Process the waypoints to be used by the geometric low level controller.
        Args:
            waypoint: The waypoints to be processed.
        Returns:
            The processed external forces to be applied to the rotors."""
        self._waypoints = waypoints
        if self._counter % self.cfg.low_level_decimation == 0:
            # eval_time = 1.0 * self._time_horizon # evaluate the spline at this timestamp
            thrusts = []
            observations = self._env.observation_manager.compute_group("policy")
            # TODO: remove [0] to generalize over all parallel envs
            drone_positions = observations[0][19:28]
            drone_orientations = observations[0][28:40]
            drone_linear_velocities = observations[0][40:49]
            drone_angular_velocities = observations[0][49:58]
            drone_linear_accelerations = observations[0][58:67]
            drone_angular_accelerations = observations[0][67:76]
            # create spline from waypoints for each drone

            for i in range(self._num_drones):
                drone_waypoints = waypoints[:, i * self._waypoint_dim * self._num_waypoints : (i + 1) * self._waypoint_dim * self._num_waypoints][0]
                drone_states: dict = {} # dict of tensors 
                drone_states["pos"] = drone_positions[i*3: i*3+3]
                drone_states["quat"] = drone_orientations[i*4: i*4+4]
                drone_states["lin_vel"] = drone_linear_velocities[i*3: i*3+3]
                drone_states["ang_vel"] = drone_angular_velocities[i*3: i*3+3]
                drone_states["lin_acc"] = drone_linear_accelerations[i*3: i*3+3]
                drone_states["ang_acc"] = drone_angular_accelerations[i*3: i*3+3]
                _, _, current_yaw = euler_xyz_from_quat(yaw_quat(drone_states["quat"].unsqueeze(0))) # TODO: remove unsqueeze
                # first_waypoint = torch.cat((drone_states["pos"], drone_states["lin_vel"], drone_states["lin_acc"], current_yaw))
                # concat first state to start of generated waypoints
                # drone_waypoints = torch.cat((first_waypoint, drone_waypoints))
                # generate spline
                # coeffs = minimum_snap_spline(drone_waypoints, self._times * self._time_horizon)
                # get individual rotor thrusts from geometric controller
                # position, velocity, acceleration, jerk, snap = evaluate_trajectory(coeffs, self._times * self._time_horizon, eval_time)
                # drone_setpoint: dict = {}
                # drone_setpoint["pos"] = position
                # drone_setpoint["lin_vel"] = velocity
                # drone_setpoint["lin_acc"] = acceleration
                # drone_setpoint["jerk"] = jerk
                # drone_setpoint["snap"] = snap
                # drone_setpoint["yaw"] = torch.tensor([0], device="cuda") # for now

                # for now, just send 1 waypoint, bypass the spline
                drone_setpoint = {}
                drone_setpoint["pos"] = drone_waypoints[:3]
                drone_setpoint["lin_vel"] = drone_waypoints[3:6]
                drone_setpoint["lin_acc"] = drone_waypoints[6:9]
                drone_setpoint["jerk"] = drone_waypoints[9:12]
                drone_setpoint["snap"] = drone_waypoints[12:15]
                drone_setpoint["yaw"] = drone_waypoints[15]

                drone_thrusts, acc_cmd, q_cmd, z_b_des = self._geometric_controller.getCommand(drone_states, self._forces[0, i*4:i*4+4], drone_setpoint)
                thrusts.append(drone_thrusts)
                if self.cfg.debug_vis:
                    self.drone_positions_debug[i] = drone_states["pos"]
                    self.drone_goals_debug[i] = drone_setpoint["pos"]
                    self.des_acc_debug[i] = acc_cmd
                    self.des_ori_debug[i] = q_cmd
                    self.z_b_debug[i] = z_b_des
            self._forces[..., 2] = torch.cat(thrusts, dim=0)
            self._counter = 0
        self._counter += 1


    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._forces = torch.clamp(self._forces, 0.0, 25.0) # TODO: change in SKRL to use sigmoid activation on last layer
            self._env.scene["robot"].set_external_force_and_torque(self._forces, self._torques, self._body_ids)

    """
    visualizations
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "drone_force_z_visualizer"):
                marker_cfg = FORCE_MARKER_Z_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_z_forces"
                self.drone_force_z_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.drone_force_z_visualizer.set_visibility(True)
            if not hasattr(self, "drone_pos_marker"):
                marker_cfg = GOAL_POS_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_pos_markers"
                self.drone_pos_marker = VisualizationMarkers(marker_cfg)
            self.drone_pos_marker.set_visibility(True)
            if not hasattr(self, "acc_marker"):
                marker_cfg = ACC_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_acc"
                self.acc_marker = VisualizationMarkers(marker_cfg)
            self.acc_marker.set_visibility(True)
            if not hasattr(self, "drone_ori_marker"):
                marker_cfg = ORIENTATION_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_ori"
                self.drone_ori_marker = VisualizationMarkers(marker_cfg)
            self.drone_ori_marker.set_visibility(True)

        else:
            if hasattr(self, "drone_force_z_visualizer"):
                self.drone_force_z_visualizer.set_visibility(False)
            if hasattr(self, "drone_pos_marker"):
                self.drone_pos_marker.set_visibility(False)
            if hasattr(self, "acc_marker"):
                self.acc_marker.set_visibility(False)
            if hasattr(self, "drone_ori_marker"):
                self.drone_ori_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # display markers

        # Get drone positions and orientations
        rotor_idx = self._robot.find_bodies("Falcon.*rotor_.*")[0]
        rotor_pos_world_frame = self._robot.data.body_state_w[:, rotor_idx, :3].view(-1, 3)
        rotor_orientation = self._robot.data.body_state_w[:, rotor_idx, 3:7].view(-1, 4)

        # marker indices for multiple envs
        marker_indices = ([0] * self._env.scene.num_envs * len(rotor_idx))
        # Rotate the arrow to point in the direction of the force
        zeros = torch.zeros(self._env.scene.num_envs, 1)
        arrow_rotation = math_utils.quat_from_euler_xyz(
            zeros, (-torch.pi / 2) * torch.ones(self._env.scene.num_envs, 1), zeros
        )  # rotate -90 degrees around y-axis in the world frame

        # Apply the offset in the drone's local frame by multiplying with drone's orientation
        arrow_rotation_offset = arrow_rotation.repeat(1, len(rotor_idx), 1).to(self.device).view(-1, 4)
        arrow_orientation = math_utils.quat_mul(rotor_orientation, arrow_rotation_offset)

        # scale arrows with applied force
        forces_to_visualize = (self._forces.view(-1, 3)).clone()
        forces_to_visualize[:, :2] += 0.5  # offset to make the forces visible
        forces_to_visualize[:, 2] /= 10  # scale down the forces
        forces_to_visualize[:, [0, 2]] = forces_to_visualize[
            :, [2, 0]
        ]  # swap around because the arrow prim is oriented in x direction
        # drone_pos_world_frame[:,2] += forces_to_visualize[:,2]/2 # offset because cylinder is drawn from the center TODO
        self.drone_force_z_visualizer.visualize(
            rotor_pos_world_frame, arrow_orientation, forces_to_visualize, marker_indices
        )

        # drone positions
        positions = torch.cat(
        (self.drone_positions_debug, self.drone_goals_debug), dim=0
        )  # visualize the payload positions in world frame
        marker_idx = [0] * 3 + [1] * 3
        self.drone_pos_marker.visualize(translations=positions, marker_indices=marker_idx)

        # drone desired accelerations

        # Normalize the desired direction vector (which represents the direction)
        acc_orientation_axis = normalize(self.des_acc_debug)
        # Define the default x-axis (the direction the arrow marker points to by default)
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self._env.scene.num_envs, self._num_drones, 1).squeeze(0)
        # Compute the dot product between x-axis and desired direction to check alignment
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        # Flip the desired direction if the dot product is negative (indicating opposite direction)
        mask = cos_angle.view(-1, 1) < 0
        acc_orientation_axis = torch.where(mask, -acc_orientation_axis, acc_orientation_axis)
        # Compute the axis of rotation (cross product between x-axis and desired direction)
        rotation_axis = torch.linalg.cross(x_axis, acc_orientation_axis)
        # Compute the angle between x-axis and desired direction using dot product
        cos_angle = torch.sum(x_axis * acc_orientation_axis, dim=-1)
        angle = torch.acos(cos_angle.clamp(-1.0, 1.0))  # Clamp to avoid numerical issues
        # Handle cases where the vectors are parallel (no rotation needed)
        rotation_axis = torch.where(
            torch.norm(rotation_axis, dim=-1, keepdim=True) < 1e-6,  # Check if parallel
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # Default to any orthogonal axis
            normalize(rotation_axis)
        )
        # Compute the quaternion from the angle-axis representation
        acc_orientation = quat_from_angle_axis(angle.view(-1), rotation_axis.view(-1, 3)).view(-1, 4)

        # Visualize the arrow marker with the new orientation
        self.acc_marker.visualize(
            self.drone_positions_debug, acc_orientation, self.des_acc_debug / 5, marker_indices=[0] * self._num_drones)
        
        # Visualize the orientation of the drone
        self.drone_ori_marker.visualize(
            self.drone_positions_debug, self.des_ori_debug, marker_indices=[0] * self._num_drones)


@configclass
class LowLevelActionCfg(ActionTermCfg):
    """Configuration for the low level action term."""

    class_type: type[ActionTerm] = LowLevelAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the scene for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset on which the forces are applied: Falcon.*base_link or Falcon.*rotor*."""
    num_drones: int = 3
    """Number of drones."""
    # waypoint_dim: int = 10
    # """Dimension of the waypoints: [pos, vel, acc, yaw]."""
    waypoint_dim: int = 16
    """Dimension of the waypoints: [pos, vel, acc, jerk, snap, yaw]."""
    num_waypoints: int = 1
    """Number of waypoints in the trajectory."""
    time_horizon: int = 2
    """Time horizon of the trajectory in seconds."""
    low_level_decimation: int = 2
    """Decimation factor for the low level action term."""
    # low_level_actions: ActionTermCfg = MISSING
    # """Low level action configuration."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
