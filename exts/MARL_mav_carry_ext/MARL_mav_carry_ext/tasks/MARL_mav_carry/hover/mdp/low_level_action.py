from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
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
        # Calculate error between waypoint and current state # TODO: figure out in which frame each term has to be
        self._high_level_action = self._high_level_action.reshape(self._env.scene.num_envs, len(self._body_ids), 4) # [force, torques]
        self._forces[...,2] = self._high_level_action[...,0]  # z force
        self._torques = self._high_level_action[...,1:]  # torques

    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        self._forces = torch.clamp(self._forces, -50.0, 50.0)
        self._torques = torch.clamp(self._torques, -0.05, 0.05)
        self._env.scene["robot"].set_external_force_and_torque(self._forces, self._torques, self._body_ids)
        # TODO: check if forces are applied to CoM

    """
    visualizations
    """

    # TODO: create markers to visualize the waypoints/trajectory and the forces/torques applied to the robot

    def _set_debug_vis_impl(self, debug_vis: bool):
    # set visibility of markers
    # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "drone_z_force_visualizer"):
                # -- display the z force on the drone
                FORCE_MARKERS = VisualizationMarkersCfg(
                markers={
                    "force_marker_1": sim_utils.CylinderCfg(
                        radius=0.01,
                        height=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.5)),
                    ),
                    "force_marker_2": sim_utils.CylinderCfg(
                        radius=0.01,
                        height=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.5)),
                    ),
                    "force_marker_3": sim_utils.CylinderCfg(
                        radius=0.01,
                        height=0.5,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.5)),
                    ),
                }
            )
                marker_cfg = FORCE_MARKERS.copy()
                marker_cfg.prim_path = "/Visuals/Actions/drone_z_forces"
                self.drone_z_force_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.drone_z_force_visualizer.set_visibility(True)
        else:
            if hasattr(self, "drone_z_force_visualizer"):
                self.drone_z_force_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # display markers
        drone_idx = self._robot.find_bodies("Falcon.*base_link")[0]
        drone_pos_world_frame = self._robot.data.body_state_w[:, drone_idx, :3].squeeze(0)
        drone_orientation = self._robot.data.body_state_w[:, drone_idx, 3:7].squeeze(0)
        forces_to_visualize = (self._forces.squeeze(0))/10
        forces_to_visualize[:, :2] += 0.5 # offset to make the forces visible
        # drone_pos_world_frame[:,2] += forces_to_visualize[:,2]/2 # offset because cylinder is drawn from the center TODO
        self.drone_z_force_visualizer.visualize(drone_pos_world_frame, drone_orientation, forces_to_visualize)

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
