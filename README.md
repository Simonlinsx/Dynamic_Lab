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
  --video-camera-preset tabletop_student_rgbd_20260712 \
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
  --video-camera-preset falling_reference_20260703 \
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
  --success-threshold 0.40 \
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
  --predicted-grasp-hold-learned-gate-threshold 0.5 \
  --predicted-grasp-hold-mode target \
  --predicted-grasp-hold-target 0.99195 0.96469 0.94568 -0.34060 -0.23838 0.98926 \
  --predicted-grasp-hold-blend 0.35 \
  --wandb-project simtoolreal_lab \
  --wandb-mode online \
  --headless \
  --device cuda:0
```

For a synchronized continuous debug video, append:

```bash
  --skip-vector-eval \
  --save-trial-sequence-videos 1 \
  --trial-sequence-trials 20 \
  --video-envs 1 \
  --video-pointcloud-visualization inset \
  --video-pointcloud-panel-resolution 320 192 \
  --video-pointcloud-range 0.30
```

`inset` places the fallback-applied RGB-D point cloud in the main video's
top-right corner. `both` additionally writes a frame-synchronized
`_pointcloud.mp4` companion.
The visualization uses the policy's current `points_palm`, RGB features, and
valid mask after temporal fallback; it never substitutes the clean simulator
point cloud when `--no-rgbd-clean-fallback` is active. `inset` is the default;
the evaluator rejects every saved student video that uses `none` or `separate`.

Formal video acceptance covers the full Revo2/Inspire x rolling/falling x
teacher/student matrix. Every cell uses a fixed third-view camera, 20 continuous
auto-reset trials, and strict post-success hold. Teacher videos show the scene;
student videos must also show the exact same-frame masked RGB-D point-cloud
observation in the top-right. `scripts/audit_teacher_student_migration.py`
checks this contract from the evaluation summary instead of trusting filenames.

The actor can also be fine-tuned with deployable RGB-D observations while the
critic receives compact simulator state. The adapter applied during rollout
must match evaluation exactly:

```bash
$PYTHON scripts/train_teacher_student_ppo.py \
  --task SimToolReal-Revo2-Franka-DynamicTabletopRollingAssetsFastSpeedAssetPrivilegedTargetHandLock-Teacher-Direct-v0 \
  --init-checkpoint outputs/teacher_student/tabletop_student_pretrain/student_pretrain_best.pt \
  --out-dir outputs/teacher_student_ppo/revo2_rolling \
  --num-envs 32 \
  --iterations 60 \
  --horizon 64 \
  --lr 5e-7 \
  --bc-anchor-weight 0.5 \
  --privileged-aux-weight 0.01 \
  --history-bootstrap zero_pad \
  --pointcloud-source rgbd_projected_mask \
  --proprio-source deployable_robot \
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

Checkpoint selection must use a separate `--first-episode-per-env` vector
evaluation. Per-iteration training success and 64-environment probes are useful
diagnostics but are not acceptance metrics.

## Current Experimental Status

The following legacy checkpoints passed the earlier July 2026 metric protocol. Teacher
success requires physical contact, catch/lift, and stable hold. Teacher videos
and student videos are continuous 20-trial sequences with automatic reset and a
fixed third view. These rows preserve historical results; they are not all exact
cross-hand comparisons under the unified protocols introduced afterward. The
legacy Revo2 rolling student video predates the mandatory point-cloud inset and
is therefore pending replacement rather than a formal video deliverable.

| Policy | Task | Vector evaluation | Strict 20-trial video |
| --- | --- | ---: | ---: |
| Revo2 teacher | rolling 0.10-0.40 m/s | 258/512 (50.39%) | 8/20 (40%) |
| Inspire teacher | rolling 50 mm sphere-only diagnostic, 0.10-0.40 m/s | 140/256 (54.69%) | 14/20 (70%) |
| Revo2 teacher | falling baton | 262/512 (51.17%) | 9/20 (45%) |
| Inspire teacher | falling baton | 293/512 (57.23%) | 12/20 (60%) |
| Revo2 RGB-D student | falling baton | 68/192 (35.42%) | 3/20 (15%; raw 6/20; point-cloud inset) |
| Revo2 RGB-D student | rolling 0.10-0.40 m/s | 82/192 (42.71%) | 4/20 (20%; raw 7/20) |

The Inspire rolling row above is a retained sphere-only diagnostic and is not
a cross-hand comparison against the Revo2 five-asset task. New comparable runs
must use `SimToolReal-Revo2-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0`
and `SimToolReal-Inspire-Franka-UnifiedRollingBenchmark-Teacher-Direct-v0`.
Both implement `rolling_multishape_v1`: the same sphere/can/bottle/cone/pill-
bottle physics and uniform evaluation distribution, a shared static-to-
0.10--0.40 m/s curriculum, 13-D action and 86-D privileged observation
contracts, reward weights, strict lift/hold success, hover target, episode
length, cameras, and direct-policy control with scripted reach/close/lift priors
disabled. Both rolling embodiments reset the Franka arm to the same upright
IsaacLab default home pose and use the same arm action scale, smoothing, and
initial target-lock duration. They also use the same home-pose-calibrated
Franka lift direction for lift-progress shaping. The lift-progress baseline is
latched after three consecutive strict-grasp steps, so home-to-object reaching
cannot be miscounted as post-grasp lift. Their seven Franka action
dimensions both use the same absolute `joint_target` interface; no embodiment
uses an incremental residual arm action in the official comparison.
The shared from-scratch reward uses dense fingertip approach plus explicit
thumb-pair/opposition shaping and penalizes non-thumb-only closure; this avoids
counting a palm scoop or four-finger push as progress toward a grasp.
For reproducible from-scratch training, use the three-stage reward curriculum:
start both hands with `SimToolReal-<Hand>-Franka-UnifiedRollingStage1-Teacher-Direct-v0`
for 300 epochs, then continue their epoch-300 checkpoints on the corresponding
`UnifiedRollingBenchmark` task. Continue checkpoints that reliably acquire a
strict grasp on `SimToolReal-<Hand>-Franka-UnifiedRollingStage3-Teacher-Direct-v0`.
Stage 1 emphasizes home-to-pregrasp reach; stage 2 emphasizes opposed
thumb-pair touch and true grasp; stage 3 makes a stationary grasp unprofitable
and rewards only object-coupled lift and stable hold while strict opposition is
maintained. Stage 3 also enables filtered fingertip-to-object contact forces:
the lift baseline latches on a sustained physical thumb-plus-two-finger grasp,
and lift, hold, and success require that force grasp to remain active. A grasp
loss is penalized once on the physical contact-to-no-contact transition after
the baseline has latched; it is not repeatedly penalized for the rest of the
episode. The shared Stage 3 weights make a prolonged no-lift grasp and unsafe
table clearance more costly than stationary contact, while object-coupled arm
lift, carry, and stable hold remain the dominant positive terms. Object
dynamics, observations, actions, success semantics, and
difficulty curriculum are unchanged at every transition, and stage 3 uses the
same reward weights for both hands.
Only embodiment-specific hand coupling, control, close posture, and the link
names/offsets behind palm and fingertip contact points remain different. Both
hands use the same table-clearance samples (Franka links plus palm and five
fingertips), normalized penalty, 3 mm tolerance, reward weight, and success
gate; the simulator still enforces each embodiment's complete collision
geometry. Evaluation reports `force_grasp_clearance_ok` so a physical grasp is
never credited through an infeasible safety proxy. Run
`scripts/check_unified_rolling_protocol.py` to verify that the shared contract
has not drifted. Teacher data collection,
student pretraining/PPO, vector evaluation, and video evaluation must reuse the
same unified task ID for the selected hand; the student script changes only the
observation source to fixed-camera masked RGB-D point clouds. It must not swap
back to a sphere-only or other legacy environment.

New falling-baton comparisons must use
`SimToolReal-Revo2-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0` and
`SimToolReal-Inspire-Franka-UnifiedFallingBatonBenchmark-Teacher-Direct-v0`.
Both implement `falling_baton_affordance_v1`: no table, the same 18 x 18 x
165 mm red/green baton, spawn-height/orientation/velocity curriculum, reward
weights, 10 mm strict contact, thumb-plus-two-finger opposition, positive-region
success, 20-step stable hold, episode length, and fixed camera contract. The
Inspire config inherits this task definition and overrides only its RH56 URDF,
six-active-DoF adapter, close posture, and control gains. Verify it with
`scripts/check_unified_falling_protocol.py`.
Falling-baton height, orientation, and velocity difficulty is gated by the
latched 20-step success rate itself; transient contact or a one-frame catch
cannot advance the curriculum.

The only allowed Revo2/Inspire differences in either protocol are the hand
URDF, six-active-DoF joint/coupling map, legal close target, hand controller
gains/smoothing, and geometry-dependent link names/offsets. Object
distributions, Franka action semantics, observations, reward/success
definitions, curriculum, clearance semantics, cameras, and evaluation seeds
are benchmark fields and must remain identical.

The current Revo2/Inspire falling rows above were produced by different legacy
curricula and therefore remain useful baselines, not the final unified
cross-hand comparison. Official unified rows are added only after vector and
fixed-camera 20-trial evaluations pass on the new task IDs.

Both student vector results use `--first-episode-per-env`: every one of the 192
initial environments contributes exactly one trial. This avoids a
completion-time bias where rapidly failing and resetting environments can be
counted repeatedly before slower successful episodes finish. Student videos
add 60 post-success steps and require physical opposition, stable relative
motion, and limited object-palm drift; `raw` is the task success before this
extra video-only hold audit.

The accepted rolling student is the success/hold-balanced epoch-6 checkpoint
with a learned hold gate in front of the calibrated six-DoF hand target. On the
64-environment calibration set, the gate reduced pre-strict-grasp hold latches
from 100% to 53.1%; on the unbiased 192-environment set it improved success
from 77/192 to 82/192. Bottle (27.8%) and can (32.6%) remain the hardest rolling
assets. Its fixed seed-119 video uses the same protocol as the previous rolling
student and converts 4/7 raw catches into a full 60-step strict hold. Seed 118
is retained as a robustness diagnostic at 1/20 strict (6/20 raw), rather than
being hidden from the experiment record. The accepted falling student remains
the original deployable checkpoint; its main gap is post-catch load-bearing
stability.

The July 12 asymmetric-PPO run is retained as an ablation, not as the accepted
student. Rolling PPO briefly reached 34/64 at iteration 15 but fell to 62/192
under formal evaluation, below the 77/192 pretrain baseline. Falling PPO peaked
at 23/64 and did not establish a meaningful improvement over the 68/192
baseline. Earlier/later hold thresholds and delayed arm-target locks also
reduced success. Increasing the rolling post-grasp hand blend from 0.35 to 0.50
or 0.70 also reduced the fixed 64-environment result from 29/64 to 24/64, so
none of these adapters are part of the final inference contract.

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
