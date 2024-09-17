from omni.isaac.lab.managers import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnv


class LowLevelAction(ActionTerm):
    """Low level action term for the hover task."""

    cfg: LowLevelActionCfg

    def __init__(self, cfg: LowLevelActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        """Process the waypoints to be used by the low level controller.
        Args:
            actions: The waypoints to be processed.
        Returns:
            The processed external forces to be applied to the rotors/falcon bodies."""
        self._torques = actions
        
    def apply_actions(self):
        """Apply the processed external forces to the rotors/falcon bodies."""
        self.env.scene.apply_external_forces(self._forces, self._torques, self._body_ids)

@configclass
class LowLevelActionCfg(ActionTermCfg):
    """Configuration for the low level action term."""
    pass


