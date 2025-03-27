import gymnasium as gym

from MARL_mav_carry_ext.tasks.managerbased.track import TrackEnvCfg

from . import agents

gym.register(
    id="Isaac-flycrane-payload-track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
