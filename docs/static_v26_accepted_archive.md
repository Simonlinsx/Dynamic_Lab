# Accepted Static Teacher Archive (v2.6)

This document identifies the immutable code and artifact contract for the
accepted Franka + Revo2/Inspire static teacher policies. The archive tag is
`static-v2.6-accepted-20260717` and the archive branch is
`archive/static-v2.6-accepted-20260717`.

The tagged commit is a complete source-tree snapshot taken immediately after
both static policies passed the deployment-oriented acceptance checks. Other
experimental task classes may exist in the same tree, but only the two task
IDs and checkpoints below belong to this accepted static contract.

## Shared contract

- Policy: privileged-state PPO actor-critic, MLP `512, 512, 256, 128`.
- Action: 13 dimensions.
- Franka: 7 bounded incremental joint targets, at most `0.015 rad` per control
  step.
- Hand: 6 absolute physical actuator targets for both Revo2 and Inspire.
- Reset: curriculum from collision-free pre-grasp to official Franka home;
  `10%` pre-grasp anchors, `50%` home anchors, and `0.02 rad` arm noise.
- Success: physical force-backed grasp, `0.10 m` lift, hover stability, and a
  continuous 180-control-step hold.
- Training: random policy initialization, 2048 environments, seed 42, PPO,
  online W&B, no scripted reach/close/lift prior, no residual policy, and no
  post-success action lock.

## Revo2

- Task: `SimToolReal-Revo2-Franka-StaticCanonicalRobustJointDelta-Teacher-Direct-v0`
- W&B: <https://wandb.ai/simonlsx/simtoolreal_lab/runs/xql0f5hc>
- Selected checkpoint: `logs/rl_games/static_canonical_v26_revo2_robusthome_jointdelta_s42_2048env_1600ep_online_20260716/nn/last_simtoolreal_dynamic_dexterous_teacher_direct_ep_1600_rew_70504.53.pth`
- Checkpoint SHA256: `253d326aa86799b3bb6a22209b03e4b0200c6f74163a8308cb67fdd196f9db7d`
- Vector evaluation: `219/256` (seed 42), `222/256` (seed 142).
- Repeated one-environment evaluation: `54/64` with a 14-second budget.
- Fixed-view video: `17/20` with a 12-second budget.
- Video: `logs/debug_videos/static/revo2_jointdelta_v26_ep1600_home_12s_20trial_20260717/20260717_072046/videos/trial_sequence_000_trials_020_success_017_sr_0.850.mp4`

## Inspire RH56BFX

- Task: `SimToolReal-Inspire-Franka-StaticCanonicalRobustJointDelta-Teacher-Direct-v0`
- W&B: <https://wandb.ai/simonlsx/simtoolreal_lab/runs/1uxwe0d4>
- Selected checkpoint: `logs/rl_games/static_canonical_v26_inspire_robusthome_jointdelta_s42_2048env_1600ep_online_20260717/nn/last_simtoolreal_dynamic_dexterous_teacher_direct_ep_450_rew_68693.73.pth`
- Checkpoint SHA256: `5d5c6b5a37d654b7bb2b47273635d7da287bc6cc9f34295b44ceec023f1d8241`
- Vector evaluation: `256/256` (seed 42), `256/256` (seed 142).
- Repeated one-environment evaluation: `64/64`.
- Fixed-view video: `20/20`.
- Video: `logs/debug_videos/static/inspire_jointdelta_v26_ep450_home_20trial_20260717/20260717_081157/videos/trial_sequence_000_trials_020_success_020_sr_1.000.mp4`

## Reproduction guard

The run-local `resolved_run_config.json` files remain beside each checkpoint
and are the authoritative PPO/config manifests. Before claiming reproduction,
verify the checkpoint SHA256 and require both a multi-environment evaluation
and a repeated single-environment fixed-view evaluation. A replicated vector
score alone is not an acceptance result.
