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
    id="SimToolReal-Revo2-Franka-DynamicTabletopSixMotorPhysicsAudit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopSixMotorPhysicsAuditEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopNativeMimicLowDrivePhysicsAudit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopNativeMimicLowDrivePhysicsAuditEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopExplicitFollowerHighDrivePhysicsAudit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopExplicitFollowerHighDrivePhysicsAuditEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopNativeSixActivePhysicsAudit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopNativeSixActivePhysicsAuditEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopOfficialSixActivePhysicsAudit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopOfficialSixActivePhysicsAuditEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopCartesian-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopCartesianTeacherEnvCfg"
        ),
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
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedPostHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedPostHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedStrictAcquisition-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedStrictAcquisitionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedAssetPrivileged-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed"
        "AssetPrivilegedTargetHandLock-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLockTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingBenchmarkTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage1-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage1TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointDeltaPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointDeltaPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianImpedancePhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianImpedancePhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStrictJointTargetAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStrictJointTargetABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStrictCartesianImpedanceAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStrictCartesianImpedanceABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverCartesianImpedanceAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverCartesianImpedanceABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetCleanCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetSynchronousContact-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetSynchronousContactTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetSynchronousCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetPostCleanLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetPostCleanLiftVelocity-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetPostCleanGripRetention-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetPostCleanActionContinuity-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetSmoothLiftPhase-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetForceBackedCleanLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetForceBackedPinchLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetForceRetainedObjectLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetSustainedForceLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticStableHoverJointTargetForceCoupledMicroLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialJointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialJointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialJointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialJointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticCanonicalJointDelta-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticCanonicalJointDeltaTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticCanonicalCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticCanonicalCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticCanonicalJointDelta-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticCanonicalJointDeltaTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticCanonicalCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticCanonicalCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticCanonicalRobustJointDelta-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticCanonicalRobustJointDeltaTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticCanonicalRobustCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticCanonicalRobustCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticCanonicalRobustJointDelta-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticCanonicalRobustJointDeltaTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticCanonicalRobustCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticCanonicalRobustCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingJointDeltaV26-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingJointDeltaV26TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingJointDeltaV26-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingJointDeltaV26TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingJointDeltaV27-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingJointDeltaV27TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingJointDeltaV27-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingJointDeltaV27TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedFallingBatonJointDeltaV26-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedFallingJointDeltaV26TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedFallingBatonJointDeltaV26-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedFallingJointDeltaV26TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereJointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereJointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereJointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereJointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereCartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereCartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereLiftStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereLiftStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereLiftStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereLiftStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereLiftStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereLiftStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereForceHoldStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereForceHoldStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereForceHoldStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereForceHoldStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereForceHoldStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereForceHoldStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereOpposedPressureHoldStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticOfficialSphereOpposedPressureHoldStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereOpposedPressureHoldStage2JointTarget-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereOpposedPressureHoldStage2JointTargetTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-StaticOfficialSphereOpposedPressureHoldStage2CartesianImpedance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireStaticOfficialSphereOpposedPressureHoldStage2CartesianImpedanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointStage2HoldPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointStage2HoldPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointDeltaStage2HoldPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointDeltaStage2HoldPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianStage2HoldPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianStage2HoldPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointDeltaFromScratchContactPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointDeltaFromScratchContactPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianFromScratchContactPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianFromScratchContactPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointDeltaStage2LiftBridgePhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointDeltaStage2LiftBridgePhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianStage2LiftBridgePhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianStage2LiftBridgePhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointStage3LiftPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointStage3LiftPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationJointDeltaStage3LiftPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceJointDeltaStage3LiftPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-StaticActionAblationCartesianStage3LiftPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2StaticActionInterfaceCartesianStage3LiftPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage2Hold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage2HoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage3-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage3TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage1CartesianPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage1AcquisitionCartesianPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Revo2-Franka-UnifiedRollingStage1AcquisitionCartesianPalmFramePhaseObs"
        "-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage1AcquisitionCartesianPalmFramePhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage2HoldCartesianPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedRollingStage3CartesianPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed"
        "AssetPrivilegedHardReplay-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedHardReplayTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed"
        "AssetPrivilegedStrictAcquisitionHardReplay-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2DynamicTabletopRollingAssetsFastSpeed"
            "AssetPrivilegedStrictAcquisitionHardReplayTeacherEnvCfg"
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
    id="SimToolReal-Revo2-Franka-FallingBatonEasyStrictAffordance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2FallingBatonEasyStrictAffordanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-FallingBatonEasyStrictAffordancePostHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2FallingBatonEasyStrictAffordancePostHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Revo2-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "Revo2UnifiedFallingBatonBenchmarkTeacherEnvCfg"
        ),
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
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftFast-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftFastDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftHoldBootstrap-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftHoldBootstrapDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarrySoftGate-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarrySoftGateDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryStreakGate-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryStreakGateDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryObjectFollow-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryObjectFollowGentle-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryObjectFollowGentleDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFix-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMediumFriction-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMediumFrictionDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFriction-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictSuccess-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictSuccessDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictMetrics-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionStrictMetricsDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionNoPrior-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionNoPriorDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHand-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandNoPrior-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixMatchedFrictionLegacyHandNoPriorDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixConservativeFrictionStrictSuccess-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixConservativeFrictionStrictSuccessDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixHighFriction-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixHighFrictionDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryHeightFixLoadBearing-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryHeightFixLoadBearingDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryCleanStart-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryCleanStartSettledLift-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartSettledLiftDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryLoadBearingLift-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryLoadBearingLiftDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryClearance-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryClearanceDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereCloseLiftCarryCleanStartConservativeFrictionStrictSuccess-DirectResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereCloseLiftCarryCleanStartConservativeFrictionStrictSuccessDirectResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereApproachBootstrap-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereApproachBootstrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereJ2ApproachBootstrap-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereJ2ApproachBootstrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80CleanStart-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80CleanStartTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80CleanStartStrictReward-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80CleanStartStrictReward"
        "FromScratch-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80CleanStartStrictRewardFromScratchTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumStrictReward"
        "FromScratch-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumStrictRewardFromScratchTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "FromScratch-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardFromScratchTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingRelativeLiftScaleProbe-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScaleProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingRelativeLiftHighScaleProbe-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftHighScaleProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingRelativeLiftScale160-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale160TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingRelativeLiftScale200PostHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingScale200PostHoldTargetHandLock-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldTargetHandLockTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingScale200PostHoldSphere60-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldSphere60TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingScale200PostHoldSphereSafeClose-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldSphereSafeCloseTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RollingScale200PostHoldMildThumbWrapClose-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldMildThumbWrapCloseTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-RollingScale200PostHoldMildThumbWrap"
        "TargetHandLock-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRollingRelativeLiftScale200PostHoldMildThumbWrapTargetHandLockTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingBenchmarkTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingStage1-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingStage1TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingStage2Hold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingStage2HoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedRollingStage3-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedRollingStage3TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuide-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideClearance-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideClearanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearance-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceThumbWrapCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceThumbWrapCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mm-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere60mm-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere60mmTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFriction-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionUnderwrap-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionUnderwrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrap-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPush-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushFixedPriorCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushFixedPriorCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative050Compat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative050Compat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StabilityCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictCarryCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictCarryCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075Streak6CarryCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075Streak6CarryCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075HybridLiftRecoveryCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075HybridLiftRecoveryCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075LiftOpenGateCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075LiftOpenGateCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StableHoldRewardOnlyCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StableHoldRewardOnlyCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbHoldCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictLiftHoldCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictLiftHoldCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictPreLiftHoldCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictPreLiftHoldCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictTimedLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictTimedLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictEarlyLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictEarlyLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairBoostCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairCurrentLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairCurrentLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLateLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLateLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCandidateProbeCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftCandidateProbeCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandMemoryCandidateProbeCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandMemoryCandidateProbeCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandAll100Compat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftHandAll100Compat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptEvalCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandMixedSpeedInterceptEvalCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandRelativeLiftUnlockedEvalCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandHighSpeedFocusInterceptCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandHighSpeedFocusInterceptCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandFastTailInterceptCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairRecentLiftEarlyHandFastTailInterceptCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairLiftHoldCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairLiftHoldCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbWrapPairBoostCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbWrapPairBoostCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictMildThumbWrapPairBoostCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictMildThumbWrapPairBoostCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairSlowLiftCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPushRelative075StrictThumbPairSlowLiftCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPriorProbeCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapLiftPriorProbeCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapRelativeLiftCandidateProbeCompat86-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionSoftUnderwrapRelativeLiftCandidateProbeCompat86TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmSoftContact-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmSoftContactTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionLiftPriorProbe-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftPriorProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionLiftCandidateProbe-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionLiftCandidateProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere50mmHighFrictionRelativeLiftCandidateProbe-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere50mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceSphere60mmHighFrictionRelativeLiftCandidateProbe-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceSphere60mmHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80FastCurriculumLooseReward"
        "LiftGuideCleanResetClearanceCanHighFrictionRelativeLiftCandidateProbe-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80FastCurriculumLooseRewardLiftGuideCleanResetClearanceCanHighFrictionRelativeLiftCandidateProbeTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80HomeSeedBootstrap-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80HomeSeedBootstrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereP80HomeSeedClearance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereP80HomeSeedClearanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereKnownGoodHomeStatic-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereKnownGoodHomeStaticTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-DynamicTabletopRollingSphereKnownGoodHomeStaticPhysical12D-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireDynamicTabletopRollingSphereKnownGoodHomeStaticPhysicalTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-PhysicalAudit-PhantomTips-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspirePhysicalAuditPhantomTipsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-PhysicalAudit-MeshOnly-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspirePhysicalAuditMeshOnlyTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-PhysicalAudit-MeshOnlyLegacyActuation-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspirePhysicalAuditMeshOnlyLegacyActuationTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-PhysicalAudit-Mimic-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspirePhysicalAuditMimicTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticJointTargetAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticJointTargetABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticCartesianImpedanceAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticCartesianImpedanceABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStrictJointTargetAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStrictJointTargetABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStrictCartesianImpedanceAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStrictCartesianImpedanceABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverCartesianImpedanceAB-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverCartesianImpedanceABTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetCleanCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetCleanCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetSynchronousContact-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetSynchronousContactTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetSynchronousCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetSynchronousCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetPostCleanLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetPostCleanLiftVelocity-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanLiftVelocityTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetPostCleanGripRetention-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanGripRetentionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetPostCleanActionContinuity-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetPostCleanActionContinuityTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetSmoothLiftPhase-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetSmoothLiftPhaseTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetForceBackedCleanLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetForceBackedCleanLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetForceBackedPinchLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetForceBackedPinchLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetForceRetainedObjectLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetForceRetainedObjectLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetSustainedForceLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetSustainedForceLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXStaticStableHoverJointTargetForceCoupledMicroLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXStaticStableHoverJointTargetForceCoupledMicroLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRolling-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage1-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage1TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage2Hold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage2HoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage3TeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage1PhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage1PhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage2HoldPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage2HoldPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3PhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage3PhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage1"
        "CartesianPhaseObs-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage1CartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage1Acquisition"
        "CartesianPhaseObs-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage1AcquisitionCartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage2Hold"
        "CartesianPhaseObs-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage2HoldCartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id=(
        "SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3"
        "CartesianPhaseObs-Teacher-Direct-v0"
    ),
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3TargetPhaseObs-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage3TargetPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedRollingStage3CartesianPriorTargetPhaseObs-Teacher-Residual-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedRollingStage3CartesianPriorTargetPhaseObsTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60RollingCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulSphere60RollingCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60ForceHold-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulSphere60ForceHoldTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60ForceLift-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulSphere60ForceLiftTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulSphere60LiftCommit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulSphere60LiftCommitTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3LiftCommit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3LiftCommitTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3LoadBearing-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3LoadBearingTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3CartesianCarry-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3CartesianCarryTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3PalmCoupledCarry-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarryTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3PalmCoupledCarry15mm-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry15mmTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3PalmCoupledCarry30mm-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3PalmCoupledCarry30mmTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3VerticalCarry15mm-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3VerticalCarry15mmTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulPrimitive3SphereBalancedLiftCommit-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulPrimitive3SphereBalancedLiftCommitTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXAnyDexRing60Bootstrap-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXAnyDexRing60BootstrapTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXAnyDexRing60ArmResidual-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXAnyDexRing60ArmResidualTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXAnyDexRing60ObjectRelative-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXAnyDexRing60ObjectRelativeTeacherEnvCfg"
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

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonStable-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_dexterous_grasp_env_cfg:InspireFallingBatonStableTeacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonStableAffordance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonStableAffordanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonEasyStableAffordance-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonEasyStableAffordanceTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonEasyCatchHoldCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonEasyCatchHoldCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonEasyPostHoldCurriculum-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonEasyPostHoldCurriculumTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonEasyPostHoldConversion-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonEasyPostHoldConversionTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireUnifiedFallingBatonBenchmarkTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedFalling-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedFallingTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedFallingCartesian-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedFallingCartesianTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-RH56BFXFaithfulUnifiedFallingMeasuredCartesian-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireRH56BFXFaithfulUnifiedFallingMeasuredCartesianTeacherEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)

gym.register(
    id="SimToolReal-Inspire-Franka-FallingBatonStableAffordanceFullSpeedEval-Teacher-Direct-v0",
    entry_point=f"{__name__}.dynamic_dexterous_grasp_env:DynamicDexterousGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.dynamic_dexterous_grasp_env_cfg:"
            "InspireFallingBatonStableAffordanceFullSpeedEvalEnvCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_teacher_ppo_cfg.yaml",
    },
)
