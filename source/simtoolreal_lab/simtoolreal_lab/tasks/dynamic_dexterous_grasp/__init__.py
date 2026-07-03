"""Gym registration for dynamic dexterous grasp teacher tasks."""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletop-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2DynamicTabletopTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopLegacyFullHand-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopLegacyFullHandTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRolling-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2DynamicTabletopRollingTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransport-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2DynamicTabletopTransportTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingStrongReward-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingStrongRewardTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportStrongReward-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportStrongRewardTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportHoldRamp-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportHoldRampTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportConservativeLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportConservativeLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportClearanceWarmup-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportClearanceWarmupTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportSkillPreserve-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportSkillPreserveTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportSkillPreserveAffordance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportSkillPreserveAffordanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportFromScratch-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportFromScratchTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportFromScratchStrictOpposition-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportFromScratchStrictOppositionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingCompat-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingCompatTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage1-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage1TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage2-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage2TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage2LiftGuide-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage2LiftGuideTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage2MotionRamp-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage2MotionRampTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsStage2MotionRamp-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsStage2MotionRampTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsStage2StableHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsStage2StableHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsStage3ResidualLiftHover-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsStage3ResidualLiftHoverTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsStage3ResidualLiftHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsStage3ResidualLiftHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage3PriorLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage3PriorLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage3ResidualLiftOnly-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftOnlyTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage3ResidualLiftHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage3ResidualLiftHoldStrictGate-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage3ResidualLiftHoldStrictGateTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallFastDirection-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallFastDirectionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallLowSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallLowSpeedEvalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallHighSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallHighSpeedEvalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallStage3TargetLiftOnly-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingSmallBallStage3TargetLiftOnlyTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportCompat-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportCompatTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingStrongRewardTargetObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingStrongRewardTargetObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingRecoveryTargetObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingRecoveryTargetObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportStrongRewardTargetObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportStrongRewardTargetObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingRichPriorTargetObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingRichPriorTargetObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopTransportRichPriorTargetObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopTransportRichPriorTargetObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2FallingBatonTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2FallingBatonFullSpeedEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonStable-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2FallingBatonStableTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonStableFullSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2FallingBatonStableFullSpeedEvalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonStableAffordance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2FallingBatonStableAffordanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonStableAffordanceFullSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2FallingBatonStableAffordanceFullSpeedEvalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonEasy-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:Revo2FallingBatonEasyTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletop-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:InspireDynamicTabletopTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidualFullSpeed-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopDirectResidualFullSpeedTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRolling-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingFastSpeed-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingFastSpeedDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingLiftFocused-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingLiftFocusedDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopTransport-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopTransportDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBaton-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:InspireFallingBatonTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:InspireFallingBatonFullSpeedEvalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)
