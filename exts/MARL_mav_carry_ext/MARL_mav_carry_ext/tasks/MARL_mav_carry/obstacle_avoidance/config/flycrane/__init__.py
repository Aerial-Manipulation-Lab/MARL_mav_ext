import gymnasium as gym

from MARL_mav_carry_ext.tasks.MARL_mav_carry.obstacle_avoidance import ObstacleEnvCfg

from . import agents

gym.register(
    id="Isaac-flycrane-payload-obstacle-avoidance-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ObstacleEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
