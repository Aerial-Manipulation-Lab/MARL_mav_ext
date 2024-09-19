import gymnasium as gym

from MARL_mav_carry_ext.tasks.MARL_mav_carry.hover.hover_env_cfg import HoverEnvCfg

gym.register(
    id="Isaac-flycrane-payload-hovering-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": HoverEnvCfg},
)
