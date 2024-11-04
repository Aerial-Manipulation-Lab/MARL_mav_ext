import gymnasium as gym

from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg import HoverEnvCfg_llc
from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover_llc.hover_env_cfg_spline import HoverEnvCfg_llc_spline

from . import agents

gym.register(
    id="Isaac-flycrane-payload-hovering-llc-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HoverEnvCfg_llc,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlycraneHoverPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-flycrane-payload-hovering-llc-spline-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HoverEnvCfg_llc_spline,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlycraneHoverPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
