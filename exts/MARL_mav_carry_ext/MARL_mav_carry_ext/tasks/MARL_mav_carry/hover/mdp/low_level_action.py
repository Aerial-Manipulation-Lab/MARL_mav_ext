from __future__ import annotations

import torch
from dataclasses import MISSING

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG
from omni.isaac.lab.utils import configclass


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
        return len(self._body_ids) * 3  # xyz coordinates

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
        # Calculate error between waypoint and current state # TODO: figure out in which frame each term has to be
        falcon_pos = self._robot.data.body_state_w[:, self._body_ids, :3]
        self._high_level_action = self._high_level_action.reshape(self._env.scene.num_envs, len(self._body_ids), 3)
        # error = self._high_level_action - falcon_pos

        # Proportional and derivative gains
        # Kp = 1.0
        # Kd = 0.1

        # Calculate forces and torques using PD controller
        # self._forces = Kp * error
        self._forces = self._high_level_action

        # self._torques = Kd * error

    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        self._forces = torch.clamp(self._forces, -20, 20.0)
        # self._torques = torch.clamp(self._torques, -10.0, 10.0)
        self._env.scene["robot"].set_external_force_and_torque(self._forces, self._torques, self._body_ids)

    """
    visualizations
    """

    # TODO: create markers to visualize the waypoints/trajectory and the forces/torques applied to the robot

    # def _set_debug_vis_impl(self, debug_vis: bool):
    # # set visibility of markers
    # # note: parent only deals with callbacks. not their visibility
    #     if debug_vis:
    #         # create markers if necessary for the first tome
    #         if not hasattr(self, "payload_pose_goal_visualizer"):
    #             # -- goal
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.prim_path = "/Visuals/Actions/payload_pose_goal"
    #             marker_cfg.markers["cuboid"].size = (0.55, 0.45, 0.15)
    #             self.payload_pose_goal_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.payload_pose_goal_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "payload_pose_goal_visualizer"):
    #             self.payload_pose_goal_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # check if robot is initialized
    #     # note: this is needed in-case the robot is de-initialized. we can't access the data
    #     if not self.robot.is_initialized:
    #         return
    #     # display markers
    #     self.payload_pose_goal_visualizer.visualize(self._env.command_manager.get_command("pose_command"))


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
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
