from dataclasses import MISSING
import torch

from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnv

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

class LowLevelAction(ActionTerm):
    """Low level action term for the hover task."""

    cfg: LowLevelActionCfg

    def __init__(self, cfg: LowLevelActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot = self.env.scene[cfg.asset_name]        
        self._high_level_action = torch.zeros(self._robot.num_instances, 3, device=self.device) # now dim is 3, xyz waypoint
        self._body_ids = self._robot.find_bodies(cfg.body_name)[0]
        self._forces = torch.zeros(self._robot.num_instances, self.action_dim, 3, device=self.device)
        self._torques = torch.zeros_like(self._forces)

    """
    properties
    """
    @property
    def action_dim(self) -> int:
        return len(self._body_ids)
    
    @property
    def high_level_action(self) -> torch.Tensor:
        return self._high_level_action
    
    @property
    def low_level_action(self) -> torch.Tensor:
        return [self._forces, self._torques]

    def process_actions(self, waypoint: torch.Tensor):
        """Process the waypoints to be used by the low level controller.
        Args:
            waypoint: The waypoints to be processed (will be trajectory later).
        Returns:
            The processed external forces to be applied to the rotors/falcon bodies."""
        self._high_level_action = waypoint

        # low level controller, for now just something random, later agilicious
        self._forces = torch.randn_like(self._forces)
        self._torques = torch.randn_like(self._torques)
        
    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        self.env.scene.apply_external_forces(self._forces, self._torques, self._body_ids)

    """
    visualizations
    """

    # TODO: create markers to visualize the waypoints/trajectory and the forces/torques applied to the robot 




