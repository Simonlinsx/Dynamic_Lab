# Teacher-Student Dynamic Dexterous Grasping Migration

Last updated: 2026-07-15

This is the IsaacLab-side contract for migrating the dynamic dexterous grasping
line from `/data1/linsixu/simtoolreal` into `simtoolreal_lab`.

## Goal

Build one teacher-student codebase for two dynamic grasp families:

1. `dynamic_tabletop_grasp`: a moving object on the table is intercepted,
   grasped, lifted, and held.
2. `falling_baton_grasp`: a baton / screwdriver-like object falls in free
   space and must be caught with the dexterous hand.

Both task families should support Franka + BrainCo Revo2 and Franka + Inspire.
The first IsaacLab route is Revo2 because that is where the IsaacGym behavior
has been hardest to stabilize.

## Framework

The intended pipeline is:

```text
privileged IsaacLab teacher env
  object pose / velocity
  clean object mesh point cloud
  clean object-local affordance labels
  palm / fingertip state
  joint state and previous targets
  contact / lift / catch / success labels
    -> privileged teacher actor-critic
    -> teacher action, value, rollout labels

deployable student observation
  RGB-D frames and object mask sequence
  masked object point-cloud sequence in palm or robot frame
  robot proprioception and previous action targets
  palm / fingertip state available on the robot
    -> temporal point-cloud perception
    -> point-flow head
    -> affordance-region head
    -> compact privileged-state prediction
    -> student actor action
```

The teacher is allowed to use simulator-only information. The student must be
able to run from deployable RGB-D/mask observations plus robot state. During RL,
we may still use an asymmetric privileged critic.

## Validated Status

Current IsaacLab RGB-D student baselines:

```text
Revo2 + tabletop:
  teacher success = 257 / 320 = 0.803125
  student success = 29 / 64 = 0.453125
  current student = success-balanced BC fine-tune + action EMA alpha 0.8
    + predicted-grasp calibrated target hand-hold adapter
    + learned residual hold correction at scale 0.25
  status = validated but still below teacher; lift-hold stability is now the bottleneck

Revo2 + falling-baton:
  teacher success = 246 / 260 = 0.946
  student success = 42 / 64 = 0.65625
  current student = aligned BC
  status = validated strong no-table aerial catch baseline

Inspire + tabletop:
  teacher success = 94 / 276 = 0.34058
  student success = 20 / 69 = 0.289855
  current student = aligned BC + one DAgger fine-tune
  status = validated; DAgger helps substantially

Inspire + falling-baton:
  teacher success = 245 / 257 = 0.953307
  student success = 53 / 64 = 0.828125
  current student = aligned BC
  status = validated strong no-table aerial catch baseline
```

## Migration Audit

Run the artifact-level audit after changing any teacher/student training,
evaluation, camera, or dataset path:

```bash
/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python scripts/audit_teacher_student_migration.py --no-strict
```

Latest audit:

```text
outputs/teacher_student_audit/20260702_024723/teacher_student_migration_audit.json
outputs/teacher_student_audit/20260702_024723/teacher_student_migration_audit.md
outputs/teacher_student_audit/20260702_024723/frames/
```

Audit status:

```text
overall_status = PASS
failed_cases = 0
warned_cases = 0
note = legacy 20260701 student summaries do not store arm_dim/hand_dim
       explicitly; the audit resolves them from metadata.action_contract via
       simtoolreal_lab.teacher_student.schema.ACTION_CONTRACTS.
```

The audit checks the four validated hand/task combinations, including
privileged teacher eval summaries, teacher rl-games TensorBoard/nn logs, RGB-D
student datasets, student pretrain checkpoints, RGB-D student eval summaries,
success videos, traces, action/proprio specs, and nonblank sample frames. The
falling-baton sample frames are also a quick visual check that the no-table
aerial catch scene is being used.

Latest training-chain audit highlights:

```text
Revo2 tabletop:
  teacher events/checkpoints = 1 / 42
  student dataset samples/success = 26624 / 170
  student best epoch/val_loss = 39 / 0.066795
Revo2 falling-baton:
  teacher events/checkpoints = 1 / 35
  student dataset samples/success = 10112 / 536
  student best epoch/val_loss = 20 / 0.246960
Inspire tabletop:
  teacher events/checkpoints = 1 / 5
  student dataset samples/success = 17664 / 22
  student best epoch/val_loss = 15 / 0.110632
Inspire falling-baton:
  teacher events/checkpoints = 1 / 14
  student dataset samples/success = 10112 / 253
  student best epoch/val_loss = 60 / 0.337128
```

## Source Mapping

The IsaacGym source of truth is still useful, but the migration should split it
into smaller IsaacLab modules instead of copying the giant task file.

Important source files in `simtoolreal`:

```text
isaacgymenvs/tasks/simtoolreal/env.py
isaacgymenvs/cfg/task/SimToolRealFallingToolV699FrankaBrainCoRevo2Front110SequentialSpindleCatch.yaml
docs/dynamic_grasp_pipeline.md
docs/v699_teacher_student_dynamic_grasp_plan.md
scripts/export_v699_student_pretrain_dataset.py
scripts/train_v699_student_pretrain.py
scripts/run_v699_teacher_student_pipeline.py
```

Target layout in `simtoolreal_lab`:

```text
simtoolreal_lab/teacher_student/schema.py
simtoolreal_lab/teacher_student/student_model.py
simtoolreal_lab/tasks/dynamic_dexterous_grasp/
  dynamic_dexterous_grasp_env.py
  dynamic_dexterous_grasp_env_cfg.py
  agents/rl_games_teacher_ppo_cfg.yaml
scripts/collect_teacher_student_dataset.py
scripts/train_teacher_student_pretrain.py
scripts/evaluate_teacher_student.py
scripts/train_teacher_student_ppo.py
scripts/audit_teacher_student_migration.py
```

## Action Contracts

Do not silently mix these interfaces:

```text
Current global deployment rule (2026-07-15):
  Revo2 and Inspire must use the same Franka action semantics.
  Candidate A = 7D measured-bounded joint absolute target.
  Candidate B = 6D base-frame measured EEF delta + torque Cartesian impedance.
  Revo2 hand = 6D physical-motor absolute target.
  Inspire hand = 6D physical-motor absolute target.
```

The static from-scratch A/B selects one Franka candidate globally. It is not
valid to retain JointTarget for one hand and Cartesian for the other. The
selected arm contract must then be used unchanged by both rolling/falling
teachers, dataset metadata, student actor outputs, evaluation, and real-robot
execution. Hand-specific motor limits and mimic kinematics remain embodiment
properties, not different policy action semantics.

Historical contracts below remain readable for artifact migration only:

```text
Revo2 semantic IsaacLab static grasp:
  13D action = 7 Franka arm targets + 6 Revo2 semantic hand fractions

Revo2 V699 active teacher/student:
  13D action = 7 Franka arm targets + 6 Revo2 active hand commands
  The IsaacLab env expands these commands to the 11 simulated hand joints
  using the official URDF mimic relations for distal joints.

Revo2 legacy V699 physical teacher/student:
  18D action = 7 Franka arm targets + 11 Revo2 physical hand joint targets

Inspire active teacher/student:
  13D action = 7 Franka arm targets + 6 Inspire active hand commands
  The IsaacLab env expands these commands to the 12 simulated Inspire joints.

Inspire legacy physical teacher/student:
  19D action = 7 Franka arm targets + 12 Inspire physical hand joint targets
```

The dynamic teacher-student route should default to the 13D active Revo2
contract so policy actions and student deployment match the real hand's six
active commands.  The old 18D V699 physical contract is legacy simulation-only
data and should not be mixed with current Revo2 training unless an explicit
adapter/distillation step is used. Inspire uses the same dataset/model schema
but switches `hand_embodiment` to `inspire` and `action_contract` to
`inspire_semantic_13d`; old `inspire_physical_19d` artifacts are legacy
simulation-only data.

For each rollout step, save both normalized policy action and raw simulator
joint-position targets.

## Dataset Contract

Student pretraining samples should contain at least:

```text
pointcloud_seq: [B, H, N, 3]
pointcloud_valid_seq: [B, H, N]
proprio_seq: [B, H, P]
target: [B, A]
point_flow_velocity or point_flow_delta: [B, N, 3]
affordance_region_labels: [B, N]
compact_privileged: [B, C]
phase: [B]
episode_success: [B]
metadata: dict
```

Recommended teacher rollout extras:

```text
obs_buf / states_buf / teacher_obs_buf
joint_pos_targets
arm_joint_target
hand_joint_target
object pose / velocity
palm pose / velocity
fingertip positions
contact / lift / catch / success labels
V699 handle-pinch metrics when using falling_baton_grasp
```

## Losses

The student objective is multi-task:

```text
L = w_rl   * PPO_or_actor_critic_loss
  + w_act  * teacher_action_distillation
  + w_val  * optional teacher_value_distillation
  + w_feat * compact_privileged_or_feature_distillation
  + w_flow * object_point_flow_loss
  + w_aff  * object_affordance_BCE
  + w_temp * temporal_consistency_loss
```

The first validated milestone is not final sim-to-real performance. It is:

1. IsaacLab environment resets and steps for both task families.
2. Teacher observations, rewards, success metrics, and videos are sane.
3. A privileged teacher can train and produce non-zero success.
4. Rollouts export the dataset schema above.
5. The student can overfit a small shard and replay reach/contact/hold behavior.
6. Student PPO fine-tuning logs reward, success, contact, lift, and videos to
   W&B online.

## Validation Gates

Every trained checkpoint should report:

```text
success_rate
contact_rate
lift_or_catch_rate
mean_episode_reward
mean_episode_length
failure breakdown if available
checkpoint path
summary JSON path
success video path
W&B run URL when enabled
```

For falling baton / V699, keep these metrics separate:

```text
physical real-contact success
geometric handle-pinch success
strict palm-grasp geometry
```

The geometric V699 signal is useful for alignment with old IK videos. The
physical contact signal is the stricter RL objective.

## 2026-07-01 Validation Snapshot

Implemented and smoke-tested IsaacLab Revo2/Inspire + Franka teacher-student
pieces:

```text
Teacher envs:
  SimToolReal-Revo2-Franka-DynamicTabletop-Teacher-Direct-v0
  SimToolReal-Revo2-Franka-FallingBaton-Teacher-Direct-v0
  SimToolReal-Revo2-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0
  SimToolReal-Inspire-Franka-DynamicTabletop-Teacher-Direct-v0
  SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidual-Teacher-Direct-v0
  SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidualFullSpeed-Teacher-Direct-v0
  SimToolReal-Inspire-Franka-FallingBaton-Teacher-Direct-v0
  SimToolReal-Inspire-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0

Student data:
  scripts/collect_teacher_student_dataset.py
  scripts/train_teacher_student_pretrain.py
  scripts/evaluate_teacher_student.py
```

Falling-baton full-speed/no-table teacher validation:

```text
checkpoint:
  logs/rl_games/revo2_falling_baton_full_no_table_from_easy_ep350_online_512env_2000ep_20260701/nn/last_simtoolreal_dynamic_dexterous_teacher_direct_ep_400_rew_1900.6381.pth
vector eval:
  outputs/eval_rl_games/falling_baton_full_speed_eval_ep400_vector_eval_online/20260701_171255/summary.json
  success_rate = 246 / 260 = 0.9461538461
success video:
  outputs/eval_rl_games/falling_baton_full_speed_ep400_success_video_wide_side_eval_online/20260701_171630/videos/success_attempt_000_env_000_idx_000.mp4
```

Tabletop teacher validation, from Franka default/home pose:

```text
ep900 checkpoint:
  logs/rl_games/revo2_dynamic_tabletop_from_home_teacher_online_512env_2000ep_20260701/nn/last_simtoolreal_dynamic_dexterous_teacher_direct_ep_900_rew_77524.11.pth
full-speed eval:
  outputs/eval_rl_games/tabletop_ep900_full_speed_vector_eval_online/20260701_174945/summary.json
  success_rate = 205 / 323 = 0.6346749226
success-video eval:
  outputs/eval_rl_games/tabletop_ep900_full_speed_success_video_online/20260701_175040/summary.json
  success_rate = 123 / 173 = 0.7109826590
success video:
  outputs/eval_rl_games/tabletop_ep900_full_speed_success_video_online/20260701_175040/videos/success_attempt_000_env_002_idx_000.mp4

best checkpoint after longer training:
  logs/rl_games/revo2_dynamic_tabletop_from_home_teacher_online_512env_2000ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
full-speed eval:
  outputs/eval_rl_games/tabletop_best_full_speed_vector_eval_online/20260701_180847/summary.json
  success_rate = 257 / 320 = 0.803125
final side-camera success-video eval:
  outputs/eval_rl_games/tabletop_best_full_speed_success_video_side_camera_online/20260701_181119/summary.json
  success_rate = 119 / 128 = 0.9296875
final side-camera success video:
  outputs/eval_rl_games/tabletop_best_full_speed_success_video_side_camera_online/20260701_181119/videos/success_attempt_000_env_003_idx_000.mp4
```

Teacher-student dataset smoke tests:

```text
tabletop dataset:
  outputs/teacher_student/tabletop_ep900_teacher_smoke_dataset.pt
  pointcloud_seq = [976, 4, 128, 3], proprio_seq = [976, 4, 91], target = [976, 18]
  action_source = teacher, task_family = dynamic_tabletop_grasp

falling-baton dataset:
  outputs/teacher_student/falling_baton_full_speed_ep400_teacher_smoke_dataset.pt
  pointcloud_seq = [976, 4, 128, 3], proprio_seq = [976, 4, 91], target = [976, 18]
  action_source = teacher, task_family = falling_baton_grasp
```

Masked RGB-D point-cloud teacher-student smoke tests:

```text
tabletop Revo2 teacher RGB-D:
  outputs/teacher_student/tabletop_best_teacher_rgbd_smoke_dataset.pt
  pointcloud_seq = [116, 4, 64, 3], proprio_seq = [116, 4, 91], target = [116, 18]
  rgbd_valid_mean = 64.0, clean_fallback_env_frames = 0
  W&B student pretrain: https://wandb.ai/simonlsx/simtoolreal_lab/runs/zaqizjwn
  val_loss: 1.527678 -> 1.064884 over 3 epochs

falling-baton Revo2 teacher RGB-D:
  outputs/teacher_student/falling_baton_full_speed_teacher_rgbd_smoke_dataset.pt
  pointcloud_seq = [116, 4, 64, 3], proprio_seq = [116, 4, 91], target = [116, 18]
  rgbd_valid_mean = 63.5, clean_fallback_env_frames = 0
  W&B student pretrain: https://wandb.ai/simonlsx/simtoolreal_lab/runs/hik08m4d
  val_loss: 1.493571 -> 1.091003 over 3 epochs
```

Revo2 tabletop full-speed RGB-D aligned student baseline:

```text
teacher checkpoint:
  logs/rl_games/revo2_dynamic_tabletop_from_home_teacher_online_512env_2000ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
task:
  SimToolReal-Revo2-Franka-DynamicTabletop-Teacher-Direct-v0

aligned teacher dataset:
  outputs/teacher_student/revo2_tabletop_best_fullspeed_teacher_rgbd_aligned_32env320_dataset.pt
  samples = 10112
  sample_timing = pre_action
  dynamic_curriculum_alpha = 1.0
  action_contract = revo2_physical_18d (legacy 18D; retrain/export current Revo2 as revo2_semantic_13d)
  pointcloud_seq = [10112, 4, 64, 3], proprio_seq = [10112, 4, 91], target = [10112, 18]
  rgbd_valid_mean_before_fallback = 63.7119, fallback_env_frames = 9
  episode_success samples = 69 / 10112

BC student:
  outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_32env320_60ep/
  best_epoch = 60
  best val_loss = 0.513440
  best val_action = 0.049826
full-speed closed-loop eval:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_vector_max420/20260701_221009/summary.json
  success_rate = 13 / 84 = 0.154762
  true_grasp_episode_rate = 0.892857
  lifted_episode_rate = 0.630952
  stable_hold_episode_rate = 0.488095
  mean_step_reward = 68.0002
  rgbd_valid_mean = 63.1099

action clamp sweep:
  clamp = 0.8:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_clamp08_vector_max420/20260701_222040/summary.json
    success_rate = 13 / 85 = 0.152941
  clamp = 0.6:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_clamp06_vector_max420/20260701_222038/summary.json
    success_rate = 4 / 64 = 0.0625
  selected inference clamp = 1.0

action EMA smoothing sweep:
  alpha = 0.5:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema05_vector_max420/20260701_223301/summary.json
    success_rate = 4 / 64 = 0.0625
  alpha = 0.7:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema07_vector_max420/20260701_223313/summary.json
    success_rate = 16 / 89 = 0.179775
  alpha = 0.8:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema08_vector_max420/20260701_223518/summary.json
    success_rate = 17 / 64 = 0.265625
    true_grasp_episode_rate = 1.0
    lifted_episode_rate = 0.609375
    stable_hold_episode_rate = 0.484375
  alpha = 0.9:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema09_vector_max420/20260701_223515/summary.json
    success_rate = 18 / 80 = 0.225
  selected inference adapter = action EMA alpha 0.8, action_rate_limit = 0.0

fixed-count vector eval note:
  scripts/evaluate_teacher_student.py now truncates the final done batch so
  requested vector-eval episode counts are exact. Older exploratory evals can
  overshoot when several envs finish on the final step. Use the fixed64 runs
  below for direct checkpoint comparison.

fixed64 baseline eval, aligned BC + action EMA alpha 0.8:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema08_vector_max420_fixed64/20260701_225444/summary.json
  success_rate = 10 / 64 = 0.15625
  true_grasp_episode_rate = 1.0
  lifted_episode_rate = 0.65625
  stable_hold_episode_rate = 0.40625

success-balanced replay fine-tune:
  checkpoint:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_success_balanced_sampler64_lr5e5_30ep/student_pretrain_best.pt
  init checkpoint:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_32env320_60ep/student_pretrain_best.pt
  sampler:
    train-sampler-label = success
    train-positive-sample-weight = 64
    action-sample-weight-mode = success
    action-positive-weight = 4
    lr = 5e-5, epochs = 30
    train positives = 65 / 9101
    expected positive fraction per epoch = 0.315
  fixed64 eval with action EMA alpha 0.8:
    outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_success_balanced_sampler64_lr5e5_30ep_ema08_vector_max420_fixed64/20260701_225447/summary.json
    success_rate = 21 / 64 = 0.328125
    true_grasp_episode_rate = 1.0
    lifted_episode_rate = 0.671875
    stable_hold_episode_rate = 0.515625

student PPO / privileged critic v0:
  script:
    scripts/train_teacher_student_ppo.py
  actor:
    PointTemporalStudent initialized from the success-balanced checkpoint
  critic:
    simulator compact_privileged state, privileged-only value function
  rollout source:
    RGB-D projected masked point cloud + robot proprioception
  implemented safeguards:
    fixed-count eval truncation
    rollout heartbeat/progress JSONL
    PPO ratio log clamp
    local metrics.jsonl and checkpoint output
  smoke:
    outputs/teacher_student_ppo/revo2_tabletop_student_ppo_smoke/20260701_230206/
    completed 8-env rollout/update and wrote student_ppo_best.pt
  short run:
    outputs/teacher_student_ppo/revo2_tabletop_success_balanced_student_ppo_privcritic_64env64h_10iter_lr2e6_ratiofix/20260701_230712/
    num_envs = 64, horizon = 64, iterations = 10, lr = 2e-6
    init_log_std = -3.0, reward_scale = 0.01
  fixed64 eval of PPO best with action EMA alpha 0.8:
    outputs/eval_teacher_student/revo2_tabletop_student_ppo_privcritic_64env64h_10iter_best_ema08_vector_max420_fixed64/20260701_231130/summary.json
    success_rate = 20 / 64 = 0.3125
    true_grasp_episode_rate = 1.0
    lifted_episode_rate = 0.734375
    stable_hold_episode_rate = 0.625
  anchored short run:
    outputs/teacher_student_ppo/revo2_tabletop_student_ppo_anchor_debug_64env64h_3iter/20260701_231454/
    lr = 5e-7, bc_anchor_weight = 0.5, ppo_ratio_limit = 2.0
    losses are finite and stable, but closed-loop eval is lower:
    outputs/eval_teacher_student/revo2_tabletop_student_ppo_anchor_debug_best_ema08_vector_max420_fixed64/20260701_231710/summary.json
    success_rate = 14 / 64 = 0.21875
    lifted_episode_rate = 0.703125
    stable_hold_episode_rate = 0.515625
  GAE mask fix:
    scripts/train_teacher_student_ppo.py now uses done[t], not done[t+1],
    when bootstrapping transition t in GAE, and logs raw advantage/return
    statistics.
  GAE-fix anchored probe:
    outputs/teacher_student_ppo/revo2_tabletop_student_ppo_gaefix_anchor_64env128h_3iter/20260701_234020/
    horizon = 128, lr = 2e-7, bc_anchor_weight = 1.0, ppo_ratio_limit = 1.5
    best fixed64 eval:
      outputs/eval_teacher_student/revo2_tabletop_student_ppo_gaefix_anchor_64env128h_3iter_best_ema08_vector_max420_fixed64/20260701_234236/summary.json
      success_rate = 12 / 64 = 0.1875
      lifted_episode_rate = 0.65625
      stable_hold_episode_rate = 0.375
  conclusion:
    PPO v0 is wired and trainable but does not replace the success-balanced
    checkpoint yet. It improves lift/stable rates but slightly lowers final
    success versus 21/64. Anchored PPO avoids the exploding-loss behavior but
    still degrades final success. Next PPO version should normalize value
    targets, keep stronger BC/action KL regularization, and use longer horizons
    or timeout-balanced episode accounting.

student inference / data diagnostics after the current best:
  clean point-cloud probe:
    outputs/eval_teacher_student/revo2_tabletop_success_balanced_cleanpc_probe_ema08_vector_max420_fixed64/20260701_232323/summary.json
    success_rate = 13 / 64 = 0.203125
    lifted_episode_rate = 0.71875
    stable_hold_episode_rate = 0.515625
    conclusion = RGB-D mask noise is not the only bottleneck.
  action rate limit probe:
    outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_rate005_vector_max420_fixed64/20260701_232421/summary.json
    action_rate_limit = 0.05, action_ema_alpha = 0.8
    success_rate = 0 / 64 = 0.0
    conclusion = uniform per-step rate limiting is too restrictive.
  EMA alpha 0.7 probe:
    outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema07_vector_max420_fixed64/20260701_232619/summary.json
    success_rate = 13 / 64 = 0.203125
    conclusion = alpha 0.8 remains better than stronger smoothing.
  slower tabletop curriculum probe:
    outputs/eval_teacher_student/revo2_tabletop_success_balanced_alpha05_ema08_vector_max420_fixed64/20260701_233501/summary.json
    dynamic_curriculum_alpha = 0.5
    success_rate = 19 / 64 = 0.296875
    lifted_episode_rate = 0.734375
    stable_hold_episode_rate = 0.546875
    conclusion = object speed contributes, but it is not the sole bottleneck.
  predicted-grasp hand-hold reflex:
    script support:
      scripts/evaluate_teacher_student.py --predicted-grasp-hold-threshold
    mechanism:
      use the student's own predicted true_grasp/grasp_seen privileged outputs
      to latch a deployable hand-hold adapter. Two modes are implemented:
      mode=max applies max(current hand action, previous hand action) on hand
      dims 7:, while mode=target softly blends hand dims toward a calibrated
      Revo2 hold target.
    naive max eval:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_predgrasp_hold09_vector_max420_fixed64/20260701_234729/summary.json
      threshold = 0.9
      success_rate = 1 / 64 = 0.015625
      lifted_episode_rate = 0.703125
      stable_hold_episode_rate = 0.484375
    conclusion = this naive max-hand hold should not be used; Revo2 hand
      closure is not safely represented by per-joint max in normalized action
      space.
  calibrated target hand-hold adapter:
    target source:
      mean hand action from the expanded teacher dataset on true_grasp samples
      with object_palm_rel_vel < 0.2
    target:
      [0.858, -0.429, -0.990, -0.975, 0.992, 0.359, 0.949, 0.964, 1.000, 0.414, 0.048]
    selected eval:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_predgrasp_targethold09_blend035_vector_max420_fixed64/20260701_235529/summary.json
      threshold = 0.9, mode = target, blend = 0.35
      success_rate = 24 / 64 = 0.375
      true_grasp_episode_rate = 1.0
      lifted_episode_rate = 0.796875
      stable_hold_episode_rate = 0.625
      mean_step_reward = 79.707
    blend probe:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_predgrasp_targethold09_blend05_vector_max420_fixed64/20260701_235714/summary.json
      threshold = 0.9, mode = target, blend = 0.5
      success_rate = 23 / 64 = 0.359375
    conclusion = calibrated soft target hold was the first strong Revo2 tabletop
      deployable student inference adapter. It improves the fixed64 baseline
      from 21/64 to 24/64 mainly through better lift and stable-hold rates.
  learned hold/reflex head v0:
    code support:
      source/simtoolreal_lab/simtoolreal_lab/teacher_student/student_model.py
        adds hold_target and hold_logits outputs
      source/simtoolreal_lab/simtoolreal_lab/teacher_student/schema.py
        adds optional hold_target and hold_mask tensors
      scripts/collect_teacher_student_dataset.py
        exports hold_target/hold_mask; default hold label is
        true_grasp && object_palm_rel_vel < 0.2
      scripts/train_teacher_student_pretrain.py
        supports --hold-loss-weight, --hold-gate-loss-weight, and
        --train-hold-head-only; also supports residual targets with
        --hold-target-mode residual and --hold-anchor-target
      scripts/evaluate_teacher_student.py
        supports --predicted-grasp-hold-mode learned_target and
        target_plus_learned_residual, plus learned gate, learned-target
        clamping, residual clamping, and residual scaling
    train:
      outputs/teacher_student/revo2_tabletop_success_balanced_learned_hold_head_64env420_lr1e3_40ep/student_pretrain_best.pt
      init checkpoint = success-balanced student
      dataset = revo2_tabletop_best_fullspeed_teacher_rgbd_aligned_64env420_dataset.pt
      hold positives = 4285 / 26624
      trained parameters = hold_head + hold_gate_head only
      best_epoch = 33
      best val_loss = 0.066963
    output diagnosis:
      positive hold_target mean is close to the calibrated target, but
      un-gated negative/transition predictions can be extreme. One sampled
      diagnostic saw learned target min/max = -14.91 / 3.73, so learned_target
      inference must clamp targets and use a learned gate.
    learned_target evals:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_learned_hold_head_blend035_vector_max420_fixed64/20260702_002217/summary.json
        mode = learned_target, blend = 0.35, no learned gate, no effective target clamp
        success_rate = 0 / 64 = 0.0
        lifted_episode_rate = 0.375
        stable_hold_episode_rate = 0.28125
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_learned_hold_head_gate05_clamp1_blend035_vector_max420_fixed64/20260702_002526/summary.json
        mode = learned_target, blend = 0.35, learned_gate_threshold = 0.5, target_clamp = 1.0
        success_rate = 16 / 64 = 0.25
        lifted_episode_rate = 0.65625
        stable_hold_episode_rate = 0.5625
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_learned_hold_head_gate05_clamp1_blend01_vector_max420_fixed64/20260702_002729/summary.json
        mode = learned_target, blend = 0.1, learned_gate_threshold = 0.5, target_clamp = 1.0
        success_rate = 18 / 64 = 0.28125
        lifted_episode_rate = 0.6875
        stable_hold_episode_rate = 0.578125
    learned gate + calibrated target eval:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_calibrated_target_learned_gate05_blend035_vector_max420_fixed64/20260702_002937/summary.json
      mode = target, calibrated hand target, blend = 0.35, learned_gate_threshold = 0.5
      success_rate = 21 / 64 = 0.328125
      lifted_episode_rate = 0.671875
      stable_hold_episode_rate = 0.546875
    conclusion = learned hold/reflex v0 is wired, trainable, and safe when
      gated/clamped, but it does not beat the calibrated target adapter. The
      absolute-target variant should not replace the calibrated target adapter.
  residual hold/reflex head v1:
    train:
      outputs/teacher_student/revo2_tabletop_success_balanced_residual_hold_head_64env420_lr1e3_40ep/student_pretrain_best.pt
      init checkpoint = success-balanced student
      dataset = revo2_tabletop_best_fullspeed_teacher_rgbd_aligned_64env420_dataset.pt
      hold positives = 4285 / 26624
      trained parameters = hold_head + hold_gate_head only
      target mode = residual around calibrated target
      anchor target:
        [0.858, -0.429, -0.990, -0.975, 0.992, 0.359, 0.949, 0.964, 1.000, 0.414, 0.048]
      best_epoch = 39
      best val_loss = 0.066795
    eval, residual scale 1.0:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_residual_hold_head_predpriv_gate_blend035_resclamp025_vector_max420_fixed64_rerun/20260702_004624/summary.json
      mode = target_plus_learned_residual
      threshold = 0.9, blend = 0.35, residual_clamp = 0.25, residual_scale = 1.0
      success_rate = 16 / 64 = 0.25
      true_grasp_episode_rate = 1.0
      lifted_episode_rate = 0.78125
      stable_hold_episode_rate = 0.671875
    eval, residual scale 0.25:
      outputs/eval_teacher_student/revo2_tabletop_success_balanced_residual_hold_head_predpriv_gate_blend035_resclamp025_scale025_vector_max420_fixed64/20260702_004834/summary.json
      mode = target_plus_learned_residual
      threshold = 0.9, blend = 0.35, residual_clamp = 0.25, residual_scale = 0.25
      success_rate = 29 / 64 = 0.453125
      true_grasp_episode_rate = 1.0
      lifted_episode_rate = 0.734375
      stable_hold_episode_rate = 0.640625
      mean_step_reward = 76.468
    conclusion = this is the current best Revo2 tabletop RGB-D student. The
      small learned residual improves over the calibrated target adapter from
      24/64 to 29/64, while a full-strength residual over-corrects and reduces
      success despite good lift/hold rates.

student success video:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_success_video_isaacgym_side/20260701_222257/videos/success_attempt_000_env_005_idx_000.mp4
student success trace:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_success_video_isaacgym_side/20260701_222257/videos/success_attempt_000_env_005_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.42]
  camera_track_target_offset = [0.0, 0.0, 0.04]
  first_lifted_step = 28
  first_stable_hold_step = 28
  first_success_step = 56
  max_object_height_delta = 0.315019

student success video with action EMA alpha 0.8:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema08_success_video_isaacgym_side_v2/20260701_224030/videos/success_attempt_000_env_013_idx_000.mp4
student success trace with action EMA alpha 0.8:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_bc_aligned_ema08_success_video_isaacgym_side_v2/20260701_224030/videos/success_attempt_000_env_013_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.42]
  camera_track_target_offset = [0.0, 0.0, 0.04]
  first_lifted_step = 34
  first_stable_hold_step = 34
  first_success_step = 263
  max_object_height_delta = 0.345933

student success video with success-balanced replay + action EMA alpha 0.8:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_success_balanced_sampler64_lr5e5_30ep_ema08_success_video_isaacgym_side/20260701_225646/videos/success_attempt_000_env_003_idx_000.mp4
student success trace with success-balanced replay + action EMA alpha 0.8:
  outputs/eval_teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_success_balanced_sampler64_lr5e5_30ep_ema08_success_video_isaacgym_side/20260701_225646/videos/success_attempt_000_env_003_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.42]
  camera_track_target_offset = [0.0, 0.0, 0.04]
  first_lifted_step = 34
  first_stable_hold_step = 37
  first_success_step = 46
  max_object_height_delta = 0.431727

student success video with calibrated target hold:
  outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_predgrasp_targethold09_blend035_success_video_isaacgym_side/20260702_000406/videos/success_attempt_000_env_001_idx_000.mp4
student success trace with calibrated target hold:
  outputs/eval_teacher_student/revo2_tabletop_success_balanced_ema08_predgrasp_targethold09_blend035_success_video_isaacgym_side/20260702_000406/videos/success_attempt_000_env_001_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.42]
  camera_track_target_offset = [0.0, 0.0, 0.04]
  first_lifted_step = 22
  first_stable_hold_step = 23
  first_success_step = 71
  max_object_height_delta = 0.243245
  video = 960x544, 68 frames, IsaacGym-aligned side view

student success video with residual target hold scale 0.25:
  outputs/eval_teacher_student/revo2_tabletop_success_balanced_residual_hold_head_scale025_success_video_isaacgym_side_16env_attempt0/20260702_010516/videos/success_attempt_000_env_008_idx_000.mp4
student success trace with residual target hold scale 0.25:
  outputs/eval_teacher_student/revo2_tabletop_success_balanced_residual_hold_head_scale025_success_video_isaacgym_side_16env_attempt0/20260702_010516/videos/success_attempt_000_env_008_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.42]
  camera_track_target_offset = [0.0, 0.0, 0.04]
  first_lifted_step = 20
  first_stable_hold_step = 21
  first_success_step = 57
  max_object_height_delta = 0.265430
  video = 960x544, 83 frames, IsaacGym-aligned side view

DAgger attempts:
  student-exec dataset:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_dagger_aligned_student_exec_32env300_dataset.pt
    samples = 9472
    teacher_student_action_l1 = 0.556277
    rgbd_valid_mean = 0.983847 valid fraction
    episode_success samples = 32 / 9472
  merged dataset:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_teacher_aligned_plus_dagger_19584_dataset.pt
    total samples = 19584
  full DAgger fine-tune:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_dagger_ft_aligned_plus_dagger_40ep/
    eval success_rate = 1 / 65 = 0.0153846
  conservative success-weighted action-only DAgger:
    outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_dagger_success_weighted_actiononly_lr5e5_25ep/
    eval success_rate = 7 / 64 = 0.109375
  action-head-only DAgger fine-tune:
    script support:
      scripts/train_teacher_student_pretrain.py --train-action-head-only
    run:
      outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_dagger_action_head_success_sampler64_lr1e4_30ep/
    eval:
      outputs/eval_teacher_student/revo2_tabletop_dagger_action_head_success_sampler64_lr1e4_30ep_ema08_vector_max420_fixed64/20260701_232131/summary.json
      success_rate = 0 / 64 = 0.0
      lifted_episode_rate = 0.203125
      stable_hold_episode_rate = 0.109375
    conclusion = lowering offline action loss on DAgger failure states can
      destroy lift/hold; do not keep pushing this variant.
  expanded clean teacher data:
    dataset:
      outputs/teacher_student/revo2_tabletop_best_fullspeed_teacher_rgbd_aligned_64env420_dataset.pt
      samples = 26624
      success samples = 170
      true_grasp samples = 23557
      rgbd valid fraction = 0.990779
    fine-tune:
      outputs/teacher_student/revo2_tabletop_best_fullspeed_rgbd_student_teacher64env420_success_balanced_lr5e5_40ep/
    eval:
      outputs/eval_teacher_student/revo2_tabletop_teacher64env420_success_balanced_lr5e5_40ep_ema08_vector_max420_fixed64/20260701_233153/summary.json
      success_rate = 9 / 64 = 0.140625
      lifted_episode_rate = 0.546875
      stable_hold_episode_rate = 0.390625
    conclusion = more full-speed teacher BC data alone did not help; it made
      the closed-loop policy more conservative and reduced lift/hold.
note:
  The current validated Revo2 tabletop RGB-D student checkpoint is the
  success-balanced fine-tune on top of the aligned BC model, with action EMA
  alpha 0.8 at inference plus the calibrated predicted-grasp target hand-hold
  adapter and a small learned residual hold correction. Action jitter is a real
  failure mode, and success samples are extremely sparse in the teacher
  dataset. Balanced replay improves the fixed64 closed-loop success from 10/64
  to 21/64, mostly by improving stable-hold frequency. The calibrated hold
  adapter then improves the same checkpoint to 24/64 by increasing lift and
  stable-hold rates. A residual hold head trained around the calibrated target
  and applied at residual_scale = 0.25 improves the same fixed64 eval to 29/64.
  The checkpoint reaches true_grasp reliably, but final success is still far
  below the privileged teacher. The first DAgger pass produced useful
  off-policy states but degraded lift/hold, likely because the merged set is
  dominated by failure-state supervision and has very sparse success samples.
  Extra teacher-only data, clean point-cloud eval, stronger smoothing, uniform
  rate limits, action-head-only DAgger, GAE-fix PPO probes, and a naive
  max-style predicted-grasp hand-hold reflex all failed to beat balanced replay.
  Student PPO / privileged critic v0 is now in the codebase, but the first
  short runs do not yet beat balanced replay. A first learned hold/reflex head
  and learned gate are now implemented and trainable, but fixed64 closed-loop
  evals show that direct learned absolute hand targets degrade success unless
  strongly gated and clamped. The residual variant is the first learned
  hand-hold head that improves the deployable Revo2 tabletop student. The next
  iteration should train the residual head on DAgger/student-executed stable
  states and combine it with longer-horizon student RL using value-target
  normalization plus a BC/KL anchor.
```

Revo2 falling-baton full-speed RGB-D aligned student baseline:

```text
teacher checkpoint:
  logs/rl_games/revo2_falling_baton_full_no_table_from_easy_ep350_online_512env_2000ep_20260701/nn/last_simtoolreal_dynamic_dexterous_teacher_direct_ep_400_rew_1900.6381.pth
task:
  SimToolReal-Revo2-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0

aligned teacher dataset:
  outputs/teacher_student/revo2_falling_baton_fullspeed_ep400_teacher_rgbd_aligned_32env320_dataset.pt
  samples = 10112
  sample_timing = pre_action
  action_contract = revo2_physical_18d (legacy 18D; retrain/export current Revo2 as revo2_semantic_13d)
  pointcloud_seq = [10112, 4, 64, 3], proprio_seq = [10112, 4, 91], target = [10112, 18]
  rgbd_valid_mean_before_fallback = 61.2755859, fallback_env_frames = 94
  true_grasp samples = 3788 / 10112
  grasp_seen samples = 4514 / 10112
  success samples = 536 / 10112

BC student:
  outputs/teacher_student/revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_32env320_60ep/
  stopped at epoch 20; best_epoch = 20
  best val_loss = 0.246960
  best val_action = 0.039205
full-speed closed-loop eval:
  outputs/eval_teacher_student/revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_ep20_vector_max420/20260701_220214/summary.json
  success_rate = 42 / 64 = 0.65625
  true_grasp_episode_rate = 0.78125
  stable_hold_episode_rate = 0.78125
  mean_step_reward = 38.6188
  rgbd_valid_mean = 58.9343
student success video:
  outputs/eval_teacher_student/revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_ep20_success_video_side/20260701_220308/videos/success_attempt_000_env_005_idx_000.mp4
student success trace:
  outputs/eval_teacher_student/revo2_falling_baton_fullspeed_ep400_rgbd_student_bc_aligned_ep20_success_video_side/20260701_220308/videos/success_attempt_000_env_005_idx_000.trace.json
  first_stable_hold_step = 15
  first_success_step = 18
  frame_count = 46
  camera_track_offset = [0.85, 1.05, 0.58]
note:
  This validates the no-table falling-baton RGB-D student path for Revo2.
  Unlike tabletop, a single aligned BC pass is already strong enough to produce
  a 65.6% full-speed student. DAgger remains useful for closing the gap to the
  privileged teacher, but was not required for a first validated student.
```

Inspire environment and student-interface smoke tests:

```text
tabletop Inspire clean:
  outputs/teacher_student/inspire_tabletop_zero_clean_smoke.pt
  pointcloud_seq = [10, 2, 32, 3], proprio_seq = [10, 2, 94], target = [10, 19]
  hand_embodiment = inspire, action_contract = inspire_physical_19d (legacy 19D)

falling-baton Inspire clean, no table:
  outputs/teacher_student/inspire_falling_baton_zero_clean_smoke.pt
  pointcloud_seq = [10, 2, 32, 3], proprio_seq = [10, 2, 94], target = [10, 19]
  hand_embodiment = inspire, action_contract = inspire_physical_19d (legacy 19D)

tabletop Inspire RGB-D:
  outputs/teacher_student/inspire_tabletop_zero_rgbd_smoke.pt
  pointcloud_seq = [14, 2, 32, 3], camera_rgb = [14, 72, 96, 3]
  rgbd_valid_mean = 32.0, clean_fallback_env_frames = 0
  student smoke out: outputs/teacher_student/inspire_tabletop_zero_rgbd_student_smoke/
  val_loss: 0.074124 -> 0.073707 over 2 epochs

falling-baton Inspire RGB-D, no table:
  outputs/teacher_student/inspire_falling_baton_zero_rgbd_smoke.pt
  pointcloud_seq = [14, 2, 32, 3], camera_rgb = [14, 72, 96, 3]
  rgbd_valid_mean = 32.0, clean_fallback_env_frames = 0
  student smoke out: outputs/teacher_student/inspire_falling_baton_zero_rgbd_student_smoke/
  val_loss: 0.441819 -> 0.387680 over 2 epochs
```

Inspire falling-baton RL smoke:

```text
run:
  logs/rl_games/inspire_falling_baton_teacher_smoke_online_64env_10ep_20260701/
checkpoint:
  logs/rl_games/inspire_falling_baton_teacher_smoke_online_64env_10ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
W&B train:
  https://wandb.ai/simonlsx/simtoolreal_lab/runs/7zaggjxe
eval summary:
  outputs/eval_rl_games/inspire_falling_baton_teacher_smoke_online_64env_10ep_eval/20260701_184146/summary.json
W&B eval:
  https://wandb.ai/simonlsx/simtoolreal_lab/runs/elxo0hqg
smoke result:
  rl_games train/eval path works for Inspire; 10 epochs is not enough for success
  success_rate = 0 / 67 = 0.0
```

Inspire falling-baton trained teacher validation:

```text
run:
  logs/rl_games/inspire_falling_baton_from_smoke_teacher_online_512env_600ep_20260701/
checkpoint:
  logs/rl_games/inspire_falling_baton_from_smoke_teacher_online_512env_600ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
W&B train:
  https://wandb.ai/simonlsx/simtoolreal_lab/runs/hyn7fchp
vector eval:
  outputs/eval_rl_games/inspire_falling_baton_from_smoke_teacher_600ep_vector_eval_online/20260701_185312/summary.json
  success_rate = 245 / 257 = 0.953307392996109
W&B eval:
  https://wandb.ai/simonlsx/simtoolreal_lab/runs/7lxccqnf
local success-video eval:
  outputs/eval_rl_games/inspire_falling_baton_from_smoke_teacher_600ep_success_video_side_camera_local/20260701_185427/summary.json
  success_rate = 124 / 129 = 0.9612403100775194
local success video:
  outputs/eval_rl_games/inspire_falling_baton_from_smoke_teacher_600ep_success_video_side_camera_local/20260701_185427/videos/success_attempt_000_env_003_idx_000.mp4
note:
  video upload to W&B was not used; the mp4 and trace are local workspace files
```

Inspire tabletop DirectResidual teacher validation, from Franka default/home
pose:

```text
task:
  SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidual-Teacher-Direct-v0
from-scratch run:
  logs/rl_games/inspire_tabletop_direct_residual_from_scratch_local_512env_1200ep_20260701/
promoted checkpoint:
  logs/rl_games/inspire_tabletop_direct_residual_from_scratch_local_512env_1200ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct_ep_355_rew_103139.586.pth
full-speed vector eval:
  outputs/eval_rl_games/inspire_tabletop_direct_residual_ep355_vector_eval_fullspeed_local/20260701_201519/summary.json
  success_rate = 94 / 276 = 0.3405797101
  true_grasp_episode_rate = 0.9963768116
  lifted_episode_rate = 0.3985507246
  stable_hold_episode_rate = 0.3913043478
success video:
  outputs/eval_rl_games/inspire_tabletop_direct_residual_ep355_success_video_side_camera_fullspeed_local/20260701_201619/videos/success_attempt_000_env_001_idx_000.mp4
success trace:
  outputs/eval_rl_games/inspire_tabletop_direct_residual_ep355_success_video_side_camera_fullspeed_local/20260701_201619/videos/success_attempt_000_env_001_idx_000.trace.json
```

Inspire tabletop full-speed fine-tune check:

```text
task:
  SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidualFullSpeed-Teacher-Direct-v0
run:
  logs/rl_games/inspire_tabletop_direct_residual_fullspeed_finetune_local_512env_to900_20260701/
best checkpoint:
  logs/rl_games/inspire_tabletop_direct_residual_fullspeed_finetune_local_512env_to900_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
vector eval:
  outputs/eval_rl_games/inspire_tabletop_direct_residual_fullspeed_finetune_best_vector_eval_local/20260701_203057/summary.json
  success_rate = 67 / 320 = 0.209375
  true_grasp_episode_rate = 1.0
  lifted_episode_rate = 0.646875
  stable_hold_episode_rate = 0.646875
decision:
  not promoted; it lifted and grasped more often, but strict continuous-hold
  success was worse than the ep355 DirectResidual checkpoint
```

Inspire falling-baton successful-teacher RGB-D student data:

```text
dataset:
  outputs/teacher_student/inspire_falling_baton_600ep_teacher_rgbd_smoke_dataset.pt
  pointcloud_seq = [244, 4, 64, 3], proprio_seq = [244, 4, 94], target = [244, 19]
  camera_rgb = [244, 72, 96, 3]
  hand_embodiment = inspire, action_contract = inspire_physical_19d (legacy 19D)
  action_source = teacher
  rgbd_valid_mean = 62.2421875, clean_fallback_env_frames = 0
student pretrain:
  outputs/teacher_student/inspire_falling_baton_600ep_teacher_rgbd_student_pretrain_smoke/
  val_loss: 1.497321 -> 1.252313 over 5 epochs
  val_action: 0.342107 -> 0.204060 over 5 epochs
note:
  W&B uploads for this private student dataset were not used. The script now
  supports --wandb-metrics-only, but external scalar logging was also denied by
  policy for private-derived student data. Local checkpoints are saved.
```

Inspire falling-baton full-speed RGB-D aligned student baseline:

```text
teacher checkpoint:
  logs/rl_games/inspire_falling_baton_from_smoke_teacher_online_512env_600ep_20260701/nn/simtoolreal_dynamic_dexterous_teacher_direct.pth
task:
  SimToolReal-Inspire-Franka-FallingBatonFullSpeedEval-Teacher-Direct-v0
scene:
  no table; falling-baton aerial catch

aligned teacher dataset:
  outputs/teacher_student/inspire_falling_baton_fullspeed_600ep_teacher_rgbd_aligned_32env320_dataset.pt
  samples = 10112
  sample_timing = pre_action
  dynamic_curriculum_alpha = 1.0
  action_contract = inspire_physical_19d (legacy 19D; retrain/export current Inspire as inspire_semantic_13d)
  pointcloud_seq = [10112, 4, 64, 3], proprio_seq = [10112, 4, 94], target = [10112, 19]
  rgbd_valid_mean_before_fallback = 60.8081, fallback_env_frames = 37
  episode_success samples = 253 / 10112

BC student:
  outputs/teacher_student/inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_32env320_60ep/
  best_epoch = 60
  best val_loss = 0.337128
  best val_action = 0.033458
full-speed closed-loop eval:
  outputs/eval_teacher_student/inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_vector_max420/20260701_222758/summary.json
  success_rate = 53 / 64 = 0.828125
  true_grasp_episode_rate = 0.953125
  stable_hold_episode_rate = 0.953125
  mean_step_reward = 64.8791
  rgbd_valid_mean = 59.5251

student success video:
  outputs/eval_teacher_student/inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_success_video_side/20260701_222859/videos/success_attempt_000_env_007_idx_000.mp4
student success trace:
  outputs/eval_teacher_student/inspire_falling_baton_fullspeed_600ep_rgbd_student_bc_aligned_success_video_side/20260701_222859/videos/success_attempt_000_env_007_idx_000.trace.json
  camera_track_offset = [0.38, -0.58, 0.38]
  camera_track_target_offset = [0.0, 0.0, 0.02]
  first_stable_hold_step = 13
  first_success_step = 22
  frame_count = 48
note:
  This validates the Inspire no-table falling-baton RGB-D student path. A
  single aligned BC pass reaches 82.8% full-speed success, close enough to the
  95.3% privileged teacher to serve as the first strong deployable-student
  baseline for the aerial catch task.
```

Inspire tabletop DirectResidual successful-teacher RGB-D student path:

```text
dataset:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_teacher_rgbd_smoke_dataset.pt
  pointcloud_seq = [1256, 4, 64, 3], proprio_seq = [1256, 4, 94], target = [1256, 19]
  hand_embodiment = inspire, action_contract = inspire_physical_19d (legacy 19D)
  action_source = teacher
  rgbd_valid_mean = 63.628125, clean_fallback_env_frames = 0
student pretrain:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_teacher_rgbd_student_pretrain_smoke/
  val_loss: 0.466538 -> 0.141478 over 25 epochs
  val_action: 0.254433 -> 0.028347 over 25 epochs
student rollout/eval smoke:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_ep355_rgbd_student_pretrain_smoke/20260701_203926/summary.json
  success_rate = 1 / 8 = 0.125
  true_grasp_episode_rate = 0.25
  rgbd_valid_mean = 62.5330779944
student success-video eval:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_ep355_rgbd_student_pretrain_success_video_smoke/20260701_204044/summary.json
student success video:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_ep355_rgbd_student_pretrain_success_video_smoke/20260701_204044/videos/success_attempt_000_env_002_idx_000.mp4
student success trace:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_ep355_rgbd_student_pretrain_success_video_smoke/20260701_204044/videos/success_attempt_000_env_002_idx_000.trace.json
note:
  this is only a 1.2k-sample BC smoke, not the final student RL/distillation
  training. It verifies that masked RGB-D point clouds can drive closed-loop
  student actions in IsaacLab and produce a successful rollout.
```

Inspire tabletop DirectResidual full-speed RGB-D BC baseline:

```text
code updates:
  scripts/collect_teacher_student_dataset.py
    supports --dynamic-curriculum-alpha for explicit full-speed student data
  scripts/train_teacher_student_pretrain.py
    supports --action-sample-weight-mode {none,true_grasp,grasp_seen,success}
  scripts/evaluate_teacher_student.py
    supports --skip-vector-eval for quick video-only diagnostics

full-speed dataset:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_teacher_rgbd_32env420_dataset.pt
  pointcloud_seq = [13344, 4, 64, 3], proprio_seq = [13344, 4, 94], target = [13344, 19]
  task = SimToolReal-Inspire-Franka-DynamicTabletop-DirectResidualFullSpeed-Teacher-Direct-v0
  dynamic_curriculum_alpha = 1.0
  rgbd_valid_mean = 63.3263392857, clean_fallback_env_frames = 0
  compact true_grasp samples = 8918 / 13344
  compact success samples = 12 / 13344

plain BC student:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_student_bc_32env420_60ep/
  val_loss: 0.354652 -> 0.081148 over 60 epochs
  val_action: 0.155396 -> 0.005889 over 60 epochs
full-speed closed-loop eval:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_bc_32env420_60ep_vector/20260701_204845/summary.json
  success_rate = 0 / 64 = 0.0
  true_grasp_episode_rate = 0.390625
  lifted_episode_rate = 0.015625
  stable_hold_episode_rate = 0.0

true-grasp-weighted BC student:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_student_bc_truegraspw3_32env420_40ep/
  action_sample_weight_mode = true_grasp, action_positive_weight = 3.0
  val_loss: 0.335016 -> 0.092049 over 40 epochs
  val_action: 0.133872 -> 0.007177 over 40 epochs
full-speed closed-loop eval:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_bc_truegraspw3_32env420_40ep_vector/20260701_210832/summary.json
  success_rate = 0 / 64 = 0.0
  true_grasp_episode_rate = 0.375
  lifted_episode_rate = 0.0
  stable_hold_episode_rate = 0.0

short failure rollout:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_bc_32env420_60ep_short40_failure_video/20260701_210526/videos/rollout_attempt_000_env_000.mp4
diagnosis:
  Full-speed supervised BC can imitate logged teacher actions to low action
  loss, but closed-loop errors still prevent reliable lift/hold. The next
  method step should use success-episode filtering / DAgger-style relabeling
  and then student RL fine-tuning with the privileged critic.
```

Inspire tabletop DirectResidual full-speed aligned + DAgger student baseline:

```text
key diagnosis:
  The original collector stored target action_t with post-step observation
  state_{t+1}. This creates a one-step behavior cloning lag. The collector now
  supports --sample-timing pre_action and uses it by default.

code updates:
  scripts/collect_teacher_student_dataset.py
    --sample-timing {pre_action,post_step}
    --action-source student_dagger
    --student-checkpoint / --student-action-clamp
    student_dagger executes the current student but stores teacher actions as target
    optional executed_action is saved for action-gap diagnostics
  scripts/filter_teacher_student_dataset.py
    sample-level filtering/rebalancing by compact labels:
    true_grasp, grasp_seen, success, episode_success, phase, valid point count
  scripts/merge_teacher_student_datasets.py
    concatenates compatible teacher/student datasets and records source_dataset_id
  scripts/train_teacher_student_pretrain.py
    --init-checkpoint for fine-tuning an existing PointTemporalStudent

aligned teacher dataset:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_teacher_rgbd_aligned_32env260_dataset.pt
  samples = 8192
  sample_timing = pre_action
  dynamic_curriculum_alpha = 1.0
  rgbd_valid_mean = 63.6170673077
  true_grasp samples = 6286 / 8192
  success samples = 18 / 8192

aligned BC student:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_student_bc_aligned_32env260_80ep/
  stopped at epoch 51; best_epoch = 50
  best val_loss = 0.0752556
full-speed eval:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_bc_aligned_32env260_ep50_vector_max420/20260701_213109/summary.json
  success_rate = 1 / 66 = 0.0151515
  true_grasp_episode_rate = 0.439394
  lifted_episode_rate = 0.0151515
  stable_hold_episode_rate = 0.0151515
diagnosis:
  Pre-action alignment fixes the supervision contract but plain BC is still not
  enough for full-speed closed-loop lift/hold.

DAgger correction dataset:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_student_dagger_aligned_student_exec_32env300_dataset.pt
  samples = 9472
  executed policy = aligned BC student
  target policy = privileged teacher ep355
  teacher_student_action_l1 = 0.549993
  true_grasp samples = 2183 / 9472
  success samples = 4 / 9472
  rgbd_valid_mean = 62.7625, fallback_env_frames = 111

merged DAgger training dataset:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_teacher_aligned_plus_dagger_17664_dataset.pt
  teacher_aligned samples = 8192
  student_dagger_exec samples = 9472
  total samples = 17664

DAgger fine-tuned student:
  outputs/teacher_student/inspire_tabletop_direct_residual_ep355_fullspeed_rgbd_student_dagger_ft_aligned_plus_dagger_40ep/
  init_checkpoint = aligned BC best
  stopped at epoch 15; best_epoch = 15
  best val_loss = 0.110632
full-speed eval:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_dagger_ft_aligned_plus_dagger_ep15_vector_max420/20260701_214931/summary.json
  success_rate = 20 / 69 = 0.289855
  true_grasp_episode_rate = 0.971014
  lifted_episode_rate = 0.347826
  stable_hold_episode_rate = 0.333333
  mean_step_reward = 231.366
success video:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_dagger_ft_ep15_success_video_side/20260701_215124/videos/success_attempt_000_env_003_idx_000.mp4
success trace:
  outputs/eval_teacher_student/inspire_tabletop_direct_residual_fullspeed_rgbd_student_dagger_ft_ep15_success_video_side/20260701_215124/videos/success_attempt_000_env_003_idx_000.trace.json
  first_lifted_step = 128
  first_stable_hold_step = 128
  first_success_step = 133
  max_object_height_delta = 0.0453926
  camera_track_offset = [0.75, 0.95, 0.62]
note:
  This is the first validated RGB-D student result on the full-speed tabletop
  task. It is still below the privileged teacher (about 34% vector eval for the
  same Inspire DirectResidual teacher checkpoint), but DAgger closes most of the
  gap from plain BC and gives a concrete student-training base for the next
  round: more DAgger iterations, success/lift-balanced replay, and student PPO
  or privileged-critic fine-tuning.
```

Student pretrain smoke tests, W&B online:

```text
tabletop:
  W&B: https://wandb.ai/simonlsx/simtoolreal_lab/runs/m4ivnqdn
  out: outputs/teacher_student/tabletop_ep900_student_pretrain_smoke/
  val_loss: 2.438132 -> 1.891803 over 5 epochs
  val_action: 0.262793 -> 0.136977

falling-baton:
  W&B: https://wandb.ai/simonlsx/simtoolreal_lab/runs/9hco9nxy
  out: outputs/teacher_student/falling_baton_ep400_student_pretrain_smoke/
  val_loss: 1.079327 -> 0.641934 over 5 epochs
  val_action: 0.214032 -> 0.126466
```
