import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from MARL_mav_carry_ext.assets import FLYCRANE_CFG  # isort:skip
import MARL_mav_carry_ext.tasks.MARL_mav_carry.hover.mdp as mdp

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm


# Define the scene configuration

@configclass
class CarryingSceneCfg(InteractiveSceneCfg):
    """Configuration for multi-drone lifting system"""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Drones
    robot: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="{ENV_REGEX_NS}/flycrane") #TODO: add joint constraints, either in URDF or here

# MDP settings

@configclass
class CommandsCfg:
    """Commands for the hovering task, TODO"""
    null = mdp.NullCommandCfg()

@configclass
class ActionsCfg:
    """Actions for the hovering task.
    TODO: add high level actions through RL planner
    And apply actions through low-level controller
    Configure multiple action terms for this.
    Figure out what to put here, does the hierarchical split go here or in obs?
    
    Lowest level should be external forces on the rotor/falcon bodies.
    Here create custom Cfg class that take in the cartesian coordinates/trajectory"""
    
@configclass
class ObservationsCfg:
    """Observations for the hovering task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy.
        TODO: create 1 observation group for the RL input"""

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Events for the hovering task.
    
    Resetting states on resets, disturbances, etc.
    """

@configclass
class RewardsCfg:
    """Rewards for the hovering task.

    Pose of payload, angle between strings"""
    
@configclass
class TerminationsCfg:
    """Terminal conditions for the hovering task.

    When the payload reaches a certain height, etc."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum for the hovering task."""

    pass

@configclass
class HoverEnvCfg:
    """Configuration for the hovering task."""

    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
