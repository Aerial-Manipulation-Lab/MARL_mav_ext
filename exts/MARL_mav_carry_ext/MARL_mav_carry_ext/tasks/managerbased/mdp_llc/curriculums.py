"""Custom functions that can be used to create curriculum for the flycrane environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import MARL_mav_carry_ext.tasks.managerbased.mdp_llc as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import CurriculumTermCfg as CurrTerm


def modify_obstacle_position(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    event_name: str,
    displacement: float,
    pos_error_threshold: float,
    ori_error_threshold: float,
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
    # obtain reset term settings
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


def modify_command_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    range: mdp.UniformPoseCommandGlobalCfg.Ranges,
    num_steps: int,
):
    """Curriculum that modifies the command range given a number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the term to be modified.
        range: The new range to be applied.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.command_manager.get_term(term_name).cfg
        # update term settings
        term_cfg.ranges = range
