# Dynamic Lab

IsaacLab codebase for dynamic dexterous grasping with Franka + BrainCo Revo2 and Franka + Inspire hands.

The project mirrors the earlier IsaacGym work in `simtoolreal`, but moves the simulation, privileged teacher RL, camera/video evaluation, and student-data bridge into IsaacLab. The main research direction is a teacher-student pipeline:

1. A privileged RL teacher observes simulator state and learns dynamic grasping.
2. A deployable student consumes masked RGB-D point-cloud history plus robot state.
3. The student predicts action, object point flow, affordance regions, and compact privileged targets.
4. The learned policy is evaluated with fixed third-view videos and, later, real-robot safety adapters.

## What Is Included

- IsaacLab DirectRLEnv tasks for Franka + Revo2 and Franka + Inspire.
- Six-active-DoF hand action contracts for Revo2 and Inspire.
- Privileged `rl_games` PPO teacher training and evaluation scripts.
- Dynamic tabletop grasp tasks:
  - rolling objects with free initial velocity, including ball, bottle, cylinder, and cone proxies;
  - transport objects on linear, curved, or turntable motion, using DOMINO20 and DextoolBench-style assets.
- Falling-baton aerial grasp tasks with no table and red/green positive-negative affordance regions.
- Affordance annotations and generated tabletop mesh assets under `assets/`.
- WAM-like student with four-frame masked RGB-D XYZRGB point-cloud history and robot-only proprioception.
- Joint student heads for action, object point flow, point affordance, compact privileged state, and post-grasp hold.
- Optional temporal geometry summaries and predicted-privilege-conditioned actions for shape-aware rolling grasps.
- W&B logging support for training, evaluation summaries, and videos.

Large local artifacts are intentionally excluded from Git: `logs/`, `outputs/`, videos, checkpoints, W&B cache, and cleanup folders.

## Repository Layout

```text
assets/
  affordance_labels/                 # SAM/affordance labels and analysis summaries
  generated/
    falling_baton_affordance/        # red-green baton URDF
    tabletop_affordance_meshes/      # generated URDF/OBJ assets for transport tasks
docs/
  teacher_student_migration.md       # detailed migration notes and experiment log
scripts/
  train_rl_games.py                  # privileged teacher PPO
  evaluate_rl_games.py               # success-rate eval and video export
  collect_teacher_student_dataset.py # clean or RGB-D student dataset collection
  train_teacher_student_pretrain.py  # point-cloud student pretraining
source/simtoolreal_lab/
  simtoolreal_lab/tasks/
    revo2_static_grasp/
    dynamic_dexterous_grasp/
  simtoolreal_lab/teacher_student/
```

## Setup

Use an IsaacLab-capable Python environment. On the original workstation this was:

```bash
export SIMTOOLREAL_LAB_PYTHON=/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python
```

From a fresh clone:

```bash
git clone https://github.com/Simonlinsx/Dynamic_Lab.git
cd Dynamic_Lab

export SIMTOOLREAL_LAB_ROOT=$(pwd)
export SIMTOOLREAL_ROOT=/path/to/simtoolreal
export PYTHONPATH=$SIMTOOLREAL_LAB_ROOT/source/simtoolreal_lab:$PYTHONPATH
export PYTHON=${SIMTOOLREAL_LAB_PYTHON:-python}

$PYTHON scripts/list_tasks.py
```

`SIMTOOLREAL_ROOT` is used for external robot assets that are not duplicated here:

```text
$SIMTOOLREAL_ROOT/assets/generated/franka_brainco_revo2_right/franka_brainco_revo2_right.urdf
$SIMTOOLREAL_ROOT/assets/generated/franka_brainco_revo2_right_v699/franka_brainco_revo2_right.urdf
$SIMTOOLREAL_ROOT/assets/embodiments/franka-inspire-z180/franka_inspire_z180.urdf
```

The generated baton and tabletop affordance assets used by the new IsaacLab tasks are stored in this repo under `assets/generated/`. The local `assets/urdf` symlink points back to the original `simtoolreal` asset tree and is ignored by Git.

## Task Families

Run `scripts/list_tasks.py` for the exact registry. Useful entry points:

| Family | Representative task id | Notes |
| --- | --- | --- |
| Static Revo2 sanity | `SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0` | Static ball/cube checks for robot mounting and hand control. |
| Revo2 rolling tabletop | `SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed-Teacher-Direct-v0` | Free rolling objects; target speed band includes 0.10-0.40 m/s. |
| Revo2 rolling eval | `SimToolReal-Revo2-Franka-DynamicTabletopRollingSmallBallLowSpeedEval-Teacher-Direct-v0` and `...HighSpeedEval...` | Low-speed and high-speed evaluation configs. |
| Revo2 transport tabletop | `SimToolReal-Revo2-Franka-DynamicTabletopTransportSkillPreserveAffordance-Teacher-Direct-v0` | DOMINO20/DextoolBench transport assets with affordance reward hooks. |
| Revo2 falling baton | `SimToolReal-Revo2-Franka-FallingBatonStableAffordance-Teacher-Direct-v0` | No table; red/green positive-negative baton regions. |
| Inspire rolling tabletop | `SimToolReal-Inspire-Franka-DynamicTabletopRollingFastSpeed-DirectResidual-Teacher-Direct-v0` | Inspire comparison route for rolling objects. |
| Inspire lift-focused rolling | `SimToolReal-Inspire-Franka-DynamicTabletopRollingLiftFocused-DirectResidual-Teacher-Direct-v0` | Continuation config that emphasizes lift and post-grasp stability. |
| Inspire falling baton | `SimToolReal-Inspire-Franka-FallingBaton-Teacher-Direct-v0` | Inspire aerial grasp teacher. |

## Teacher Training

Direct `rl_games` entry:

```bash
$PYTHON scripts/train_rl_games.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed-Teacher-Direct-v0 \
  --num-envs 512 \
  --seed 42 \
  --max-epochs 900 \
  --horizon-length 32 \
  --minibatch-size 8192 \
  --mini-epochs 3 \
  --save-frequency 50 \
  --experiment-name revo2_rolling_fast_teacher \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

Convenience wrapper for online dynamic teacher runs:

```bash
TASK=SimToolReal-Revo2-Franka-FallingBatonStableAffordance-Teacher-Direct-v0 \
GPU_ID=0 \
EXPERIMENT_NAME=revo2_falling_baton_affordance_teacher \
WANDB_PROJECT=simtoolreal_lab \
scripts/run_dynamic_teacher_online.sh
```

Inspire rolling comparison:

```bash
$PYTHON scripts/train_rl_games.py \
  --task SimToolReal-Inspire-Franka-DynamicTabletopRollingFastSpeed-DirectResidual-Teacher-Direct-v0 \
  --num-envs 512 \
  --seed 54 \
  --max-epochs 900 \
  --horizon-length 32 \
  --minibatch-size 8192 \
  --mini-epochs 3 \
  --experiment-name inspire_rolling_fast_teacher \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

To continue a run from a checkpoint while resetting optimizer state:

```bash
$PYTHON scripts/train_rl_games.py \
  --task SimToolReal-Inspire-Franka-DynamicTabletopRollingLiftFocused-DirectResidual-Teacher-Direct-v0 \
  --checkpoint logs/rl_games/<run>/nn/<checkpoint>.pth \
  --reset-optimizer-on-load \
  --reset-epoch-on-load \
  --num-envs 512 \
  --experiment-name inspire_rolling_lift_focused_continue \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

## Evaluation And Videos

Vector evaluation plus fixed third-view videos:

```bash
$PYTHON scripts/evaluate_rl_games.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed-Teacher-Direct-v0 \
  --checkpoint logs/rl_games/<run>/nn/<checkpoint>.pth \
  --num-envs 128 \
  --episodes 256 \
  --success-threshold 0.30 \
  --save-success-videos 2 \
  --video-attempts 40 \
  --video-envs 1 \
  --video-post-success-steps 48 \
  --video-camera-eye 0.80 1.05 0.75 \
  --video-camera-target 0.35 0.00 0.55 \
  --video-camera-resolution 1280 720 \
  --video-camera-focal-length 20 \
  --no-video-camera-track-object \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

For falling-baton debugging, save one continuous video containing 20 reset trials:

```bash
$PYTHON scripts/evaluate_rl_games.py \
  --task SimToolReal-Revo2-Franka-FallingBatonStableAffordanceFullSpeedEval-Teacher-Direct-v0 \
  --checkpoint logs/rl_games/<run>/nn/<checkpoint>.pth \
  --skip-vector-eval \
  --save-success-videos 0 \
  --save-trial-sequence-videos 1 \
  --trial-sequence-trials 20 \
  --video-envs 1 \
  --video-camera-eye 0.80 1.05 0.75 \
  --video-camera-target 0.35 0.00 0.62 \
  --video-camera-resolution 1280 720 \
  --video-camera-focal-length 20 \
  --no-video-camera-track-object \
  --headless \
  --device cuda:0
```

Evaluation writes JSON summaries, per-trial traces, and videos under `outputs/eval_rl_games/` by default. These artifacts are not committed.

## Student Data And Pretraining

Collect teacher-student data with clean simulator object point clouds:

```bash
$PYTHON scripts/collect_teacher_student_dataset.py \
  --task SimToolReal-Revo2-Franka-FallingBatonStableAffordance-Teacher-Direct-v0 \
  --out outputs/teacher_student/falling_baton_clean_pc.pt \
  --num-envs 32 \
  --steps 1024 \
  --history 4 \
  --object-points 256 \
  --headless \
  --device cuda:0
```

Collect projected RGB-D/masked point-cloud data:

```bash
$PYTHON scripts/collect_teacher_student_dataset.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeed-Teacher-Direct-v0 \
  --pointcloud-source rgbd_projected_mask \
  --out outputs/teacher_student/tabletop_rgbd_pc.pt \
  --num-envs 32 \
  --steps 752 \
  --history 4 \
  --history-bootstrap zero_pad \
  --object-points 128 \
  --point-features xyzrgb \
  --proprio-source deployable_robot \
  --sample-timing pre_action \
  --no-rgbd-clean-fallback \
  --rgbd-temporal-fallback \
  --rgbd-width 320 \
  --rgbd-height 180 \
  --rgbd-mask-points 1536 \
  --rgbd-mask-dilation 2 \
  --rgbd-depth-tolerance 0.045 \
  --student-camera-eye 1.25 -1.05 0.72 \
  --student-camera-target 0.58 0.00 0.37 \
  --student-camera-focal-length 24 \
  --headless \
  --device cuda:0
```

For DAgger, the current student executes the rollout while the privileged
teacher labels every visited state. This closes the distribution gap left by
pure teacher-rollout behavior cloning without adding simulator state to the
deployable input:

```bash
$PYTHON scripts/collect_teacher_student_dataset.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLock-Teacher-Direct-v0 \
  --checkpoint logs/rl_games/<teacher-run>/nn/<teacher-checkpoint>.pth \
  --action-source student_dagger \
  --student-checkpoint outputs/teacher_student/<student>.pt \
  --student-action-mode mean \
  --dagger-teacher-probability 0.0 \
  --pointcloud-source rgbd_projected_mask \
  --point-features xyzrgb \
  --proprio-source deployable_robot \
  --history 4 \
  --history-bootstrap zero_pad \
  --no-rgbd-clean-fallback \
  --rgbd-temporal-fallback \
  --out outputs/teacher_student/rolling_dagger.pt \
  --headless \
  --device cuda:0
```

The exported sample contains the deployable point-cloud/proprioception history,
teacher action, point flow, point affordance, compact privileged target, and
post-grasp hold labels. RGB-D masking uses each environment's active primitive
shape and physical size; it never fills a missed object mask with arbitrary
finite table or hand depth.

Pretrain the temporal point-cloud student. `--geometric-summary` exposes object
centroid, extent, and inter-frame displacement. The optional
`--privileged-action-conditioning` flag routes the student's predicted relative
state into its action head; the validated rolling checkpoint does not require
that option.

```bash
$PYTHON scripts/train_teacher_student_pretrain.py \
  --dataset outputs/teacher_student/tabletop_rgbd_pc.pt \
  --out-dir outputs/teacher_student/tabletop_student_pretrain \
  --epochs 60 \
  --batch-size 512 \
  --geometric-summary \
  --flow-target delta \
  --flow-loss smooth_l1 \
  --flow-target-max-norm 0.1 \
  --action-loss-weight 5.0 \
  --privileged-loss smooth_l1 \
  --hold-loss-weight 0.25 \
  --hold-gate-loss-weight 0.1 \
  --device cuda:0 \
  --wandb-project simtoolreal_lab \
  --wandb-mode online
```

Evaluate a deployable student from the same fixed camera used to form its RGB-D input:

```bash
$PYTHON scripts/evaluate_teacher_student.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLock-Teacher-Direct-v0 \
  --checkpoint outputs/teacher_student/tabletop_student_pretrain/student_pretrain_best.pt \
  --num-envs 192 \
  --episodes 192 \
  --first-episode-per-env \
  --success-threshold 0.20390625 \
  --pointcloud-source rgbd_projected_mask \
  --proprio-source deployable_robot \
  --history-bootstrap zero_pad \
  --rgbd-width 320 \
  --rgbd-height 180 \
  --rgbd-mask-points 1536 \
  --rgbd-mask-dilation 2 \
  --rgbd-depth-tolerance 0.045 \
  --student-camera-eye 1.25 -1.05 0.72 \
  --student-camera-target 0.58 0.00 0.37 \
  --no-rgbd-clean-fallback \
  --rgbd-temporal-fallback \
  --predicted-grasp-hold-threshold 0.5 \
  --predicted-grasp-hold-mode target \
  --predicted-grasp-hold-target 0.99195 0.96469 0.94568 -0.34060 -0.23838 0.98926 \
  --predicted-grasp-hold-blend 0.35 \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

## Current Experimental Status

The following checkpoints passed the July 2026 acceptance protocol. Teacher
success requires physical contact, catch/lift, and stable hold. Teacher videos
and student videos are continuous 20-trial sequences with automatic reset and a
fixed third view.

| Policy | Task | Vector evaluation | Strict 20-trial video |
| --- | --- | ---: | ---: |
| Revo2 teacher | rolling 0.10-0.40 m/s | 258/512 (50.39%) | 8/20 (40%) |
| Inspire teacher | rolling 0.10-0.40 m/s | 80/160 (50.00%) | 11/20 (55%) |
| Revo2 teacher | falling baton | 262/512 (51.17%) | 9/20 (45%) |
| Inspire teacher | falling baton | 293/512 (57.23%) | 12/20 (60%) |
| Revo2 RGB-D student | falling baton | 45/192 (23.44%) | 5/20 (25%) |
| Revo2 RGB-D student | rolling 0.10-0.40 m/s | 74/192 (38.54%) | 5/20 (25%) |

The rolling student vector result uses `--first-episode-per-env`: every one of
the 192 initial environments contributes exactly one trial. This avoids a
completion-time bias where rapidly failing and resetting environments can be
counted repeatedly before slower successful episodes finish.

Curated checkpoints, summaries, and videos are indexed under
`outputs/curated_artifacts/` and `outputs/curated_videos/` on the experiment
workstation. These ignored artifacts are not part of Git; W&B stores the online
metric history.

Run the lightweight artifact audit without starting IsaacLab:

```bash
$PYTHON scripts/audit_teacher_student_migration.py \
  --manifest docs/teacher_student_acceptance_20260712.json \
  --out-dir outputs/curated_artifacts/audit \
  --fail-on-warning
```

The accepted run produces `outputs/curated_artifacts/acceptance_report.md` and
`acceptance_report.json` with overall status `PASS`.

## Development Notes

- Keep code and reusable lightweight assets in Git.
- Keep experiment products in `logs/` or `outputs/`; both are ignored.
- Prefer fixed third-view video cameras for debugging and sim2real comparison.
- Use `--wandb-mode online` for synced runs, `offline` for local-only runs, and `disabled` for smoke tests.
- Check third-party asset licenses before redistributing additional robot or object mesh trees.
