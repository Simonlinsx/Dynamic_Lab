"""Gym registration for Franka + Revo2 static grasp tasks."""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-InitialPose-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallInitialPoseGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-InitialPoseLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallInitialPoseLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-InitialPoseStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallInitialPoseStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaPlannerHomeStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaPlannerHomeStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaSrdfHomeStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaSrdfHomeStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeLiftShapedStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeLiftShapedStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeReachThenLiftStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeReachThenLiftStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeHoldPriorityStrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeHoldPriorityStrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeHoldPriorityV2StrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeHoldPriorityV2StrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeHoldPriorityV3StrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeHoldPriorityV3StrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeHoldPriorityV4StrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeHoldPriorityV4StrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-FrankaDefaultHomeHoldPriorityV5StrictLift-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallFrankaDefaultHomeHoldPriorityV5StrictLiftGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticBall-ResidualGrasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticBallResidualGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticCube3cm-Grasp-Direct-v0",
    entry_point=f"{__name__}.revo2_static_grasp_env:Revo2StaticGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.revo2_static_grasp_env_cfg:Revo2StaticCube3cmGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
