from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_mul, quat_inv

from .marker_utils import FORCE_MARKER_Z_CFG, TORQUE_MARKER_CFG


class LowLevelAction(ActionTerm):
    """Low level action term for the hover task."""

    cfg: LowLevelActionCfg

    def __init__(self, cfg: LowLevelActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_ids = self._robot.find_bodies(cfg.body_name)[0]
        self._high_level_action = torch.zeros(
            len(self._body_ids) * 3, device=self.device
        )  # now dim is 3 drones * xyz waypoint
        self._forces = torch.zeros(env.scene.num_envs, len(self._body_ids), 3, device=self.device)
        self._torques = torch.zeros_like(self._forces)

    """
    properties
    """

    @property
    def action_dim(self) -> int:
        return len(self._body_ids) * 4  # for now: z force + 3 torques * 3 drones

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._high_level_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return [self._forces, self._torques]

    def process_actions(self, waypoint: torch.Tensor):
        """Process the waypoints to be used by the low level controller.
        Args:
            waypoint: The waypoints to be processed (will be trajectory later).
        Returns:
            The processed external forces to be applied to the rotors/falcon bodies."""
        self._high_level_action = waypoint  # vector with 3 waypoints for 3 drones

        # low level controller, for now just something random, later agilicious
        self._high_level_action = self._high_level_action.reshape(
            self._env.scene.num_envs, len(self._body_ids), 4
        )  # [force, torques]
        self._forces[..., 2] = self._high_level_action[..., 0]  # z force
        self._torques = self._high_level_action[..., 1:]  # torques

    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        self._forces = torch.clamp(self._forces, 0.0, 25.0)
        self._env.scene["robot"].set_external_force_and_torque(self._forces, self._torques, self._body_ids)

    """
    visualizations
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "drone_z_force_visualizer"):
                marker_cfg = FORCE_MARKER_Z_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_z_forces"
                self.drone_force_z_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "drone_torque_visualizer"):
                marker_cfg = TORQUE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_torques"
                self.drone_torque_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.drone_force_z_visualizer.set_visibility(True)
            self.drone_torque_visualizer.set_visibility(True)
        else:
            if hasattr(self, "drone_force_z_visualizer"):
                self.drone_force_z_visualizer.set_visibility(False)
            if hasattr(self, "drone_torque_visualizer"):
                self.drone_torque_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # display markers
        # marker indices for multiple envs
        marker_indices = (
            [0] * self._env.scene.num_envs + [1] * self._env.scene.num_envs + [2] * self._env.scene.num_envs
        )

        # Get drone positions and orientations
        drone_idx = self._robot.find_bodies("Falcon.*base_link")[0]
        drone_pos_world_frame = self._robot.data.body_state_w[:, drone_idx, :3].view(-1, 3)
        drone_orientation = self._robot.data.body_state_w[:, drone_idx, 3:7].view(-1, 4)

        # Rotate the arrow to point in the direction of the force
        zeros = torch.zeros(self._env.scene.num_envs, 1)
        arrow_rotation_offset = math_utils.quat_from_euler_xyz(
            zeros, (-torch.pi / 2) * torch.ones(self._env.scene.num_envs, 1), zeros
        )  # rotate -90 degrees around y-axis in the world frame

        # Apply the offset in the drone's local frame by multiplying with drone's orientation
        arrow_rotation_offset = arrow_rotation_offset.repeat(1, 3, 1).to(self.device).view(-1, 4)
        arrow_orientation = math_utils.quat_mul(drone_orientation, arrow_rotation_offset)

        # scale arrows with applied force
        forces_to_visualize = (self._forces.view(-1, 3)).clone()
        forces_to_visualize[:, :2] += 1.0  # offset to make the forces visible
        forces_to_visualize[:, 2] /= 15  # scale down the forces
        forces_to_visualize[:, [0, 2]] = forces_to_visualize[
            :, [2, 0]
        ]  # swap around because the arrow prim is oriented in x direction
        # drone_pos_world_frame[:,2] += forces_to_visualize[:,2]/2 # offset because cylinder is drawn from the center TODO
        self.drone_force_z_visualizer.visualize(
            drone_pos_world_frame, arrow_orientation, forces_to_visualize, marker_indices
        )

        # visualize torques
        torques_to_visualize = (self._torques.view(-1, 3)).clone()
        torques_to_visualize += 0.01  # offset to make the torques visible
        torques_to_visualize *= 10  # scale up the torques
        # print(torques_to_visualize)
        self.drone_torque_visualizer.visualize(
            drone_pos_world_frame, drone_orientation, torques_to_visualize, marker_indices
        )


@configclass
class LowLevelActionCfg(ActionTermCfg):
    """Configuration for the low level action term."""

    class_type: type[ActionTerm] = LowLevelAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the scene for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset on which the forces are applied: Falcon.*base_link or Falcon.*rotor*."""
    # low_level_decimation: int = 4
    # """Decimation factor for the low level action term."""
    # low_level_actions: ActionTermCfg = MISSING
    # """Low level action configuration."""
    # low_level_observations: ObservationGroupCfg = MISSING
    # """Low level observation configuration."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""
