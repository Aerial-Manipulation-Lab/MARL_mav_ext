"""Custom functions that can be used to create curriculum for the flycrane environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm

def modify_obstacle_position(
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        command_name: str, 
        event_name: str, 
        displacement: float, 
        pos_error_threshold: float,
        ori_error_threshold: float
        ) -> torch.Tensor:
    """Curriculum that modifies obstacle spawning position after reaching a certain performance.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        event_name: The name of the event term.
        displacement: The displacement to be applied to the obstacle.
        error_treshold: The error threshold to be reached before applying the displacement.
    """
    # check if metric at termination has reached a certain threshold
    # obtain resest term settings
    term_cfg = env.event_manager.get_term_cfg(event_name)

    if "log" in env.extras:
        position_error = env.extras["log"]["Metrics/" + command_name + "/position_error"]
        position_error = 0
        orientation_error = env.extras["log"]["Metrics/" + command_name + "/orientation_error"]
        orientation_error = 0
        if position_error < pos_error_threshold and orientation_error < ori_error_threshold:
            # update term settings
            y_range = list(term_cfg.params["pose_range"]["y"])
            y_range[1] += displacement
            y_range[1] = min(y_range[1], 4.0)
            term_cfg.params["pose_range"]["y"] = tuple(y_range)

    return torch.tensor([term_cfg.params["pose_range"]["y"][1]], dtype=torch.float32)