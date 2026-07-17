# Revo2 Action-Interface Ablation

This benchmark isolates the Franka arm action parameterization while keeping
the Revo2 hand, object, reset state, observation semantics, reward, contact
criteria, curriculum stage, PPO seed, and training budget fixed.

## 2026-07-15 hand-physics correction

The torque-level Cartesian comparison is paused until the hand embodiment is
identical across both arm controllers. The previous strict JointTarget task
used the URDF-native PhysX mimic constraints, while the torque-Cartesian task
used eleven independent hand joints with software-expanded follower targets.
That is a hand-physics confound, so it cannot support a Joint-versus-Cartesian
conclusion.

`mimic` is the mechanical coupling authored in the official Revo2 URDF. The
policy and real hand expose six motors, while five distal joints obey the
following constraints instead of receiving independent commands:

- thumb distal = `1.0 * thumb proximal`;
- each finger distal = `1.155 * finger proximal`.

The known simple 13-D joint-target policy was evaluated for 256 static,
zero-speed episodes in each hand-physics variant. No policy weights, object,
reset distribution, observation, arm controller, reward, or random seed were
changed. Force diagnostics use actual filtered PhysX fingertip/object contact.

| Hand physics | Legacy success | Force grasp | Force lift | Force stable hold | Dropped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native mimic, high drive, targets written to 11 joints | 40.63% | 84.38% | 80.86% | 66.80% | 19.14% |
| Independent followers, low drive | 10.55% | 77.34% | 76.17% | 34.77% | 48.83% |
| Native mimic, low drive, targets written to 11 joints | 50.00% | 67.58% | 65.63% | 46.09% | 33.59% |
| Independent followers, high drive | 1.56% | 73.83% | 64.84% | 10.55% | 59.77% |
| Native mimic, high drive, only six active targets | 35.16% | 81.64% | 76.56% | 55.08% | 28.13% |
| Native mimic, official URDF limits, only six active targets | 2.34% | 2.34% | 0.39% | 0.00% | 63.28% |

W&B runs, in table order:

- `4lstd9zo`
- `6v74kn6k`
- `ngx3gzy7`
- `w91cro6m`
- `iswk6med`
- `jwov479w`

Authoritative summaries are under:

- `outputs/eval_rl_games/revo2_known_simple_jointtarget_best_static_alpha0_force_audit_20260715/20260715_192005`
- `outputs/eval_rl_games/revo2_known_simple_checkpoint_static_sixmotor_physics_audit_20260715/20260715_192711`
- `outputs/eval_rl_games/revo2_known_simple_checkpoint_static_native_mimic_low_drive_audit_20260715/20260715_193304`
- `outputs/eval_rl_games/revo2_known_simple_checkpoint_static_explicit_follower_high_drive_audit_20260715/20260715_195142`
- `outputs/eval_rl_games/revo2_known_simple_checkpoint_static_native_six_active_audit_20260715/20260715_195346`
- `outputs/eval_rl_games/revo2_known_simple_checkpoint_static_official_six_active_audit_20260715/20260715_195617`

Observations:

1. Removing the native mimic constraint is the dominant cause of unstable
   carrying. Raising independent-follower gains does not repair it and makes
   the zero-shot policy substantially worse.
2. Driving only the six real motor joints while leaving all distal joints to
   native mimic remains physically viable. This is the deployment action
   contract for subsequent experiments.
3. The old checkpoint depends strongly on its historical high-torque hand
   drive. Its 2.34% zero-shot result under official effort limits is evidence
   of controller-distribution shift, not evidence that an official-limit hand
   cannot learn from scratch.
4. Legacy task success is not a sufficient physical metric. Distal-link force
   grasp is also not a sufficient or necessary metric by itself: a load may be
   carried through the palm or proximal links without registering on all five
   distal sensors. Final acceptance requires actual object lift, bounded
   object/palm and world velocity, hover-target tolerance, and a consecutive
   hold streak. Full-hand contact is audited separately when needed.
5. No Joint-versus-Cartesian conclusion is claimed from this table. It is a
   prerequisite hand-physics audit using one existing joint-target policy.

Next locked protocol:

- one shared six-active-motor, native-mimic Revo2 model for both controllers;
- official URDF joint, effort, and velocity limits first, with any calibrated
  relaxation reported as a separate domain-randomized profile;
- no residual, scripted close, scripted reach, lift prior, or motion planner;
- identical static object/reset, privileged observations, state-only reward,
  PPO architecture, seed set, environment count, and frame budget;
- Joint arm = seven joint targets; Cartesian arm = base-frame 6-D measured
  EEF delta with torque-level Cartesian impedance; hand = six absolute motor
  targets in both cases;
- at least 200 strict stable-hover episodes and one fixed-view 20-trial video
  before either interface is accepted; distal force traces remain diagnostic.

## 2026-07-15 official six-active runtime lock

Cross-hand deployment constraint: the final Revo2 and Inspire teachers and
students must use one global action contract. A hand-specific winner is not
allowed. The global choice is either 7-D Franka joint absolute targets or 6-D
base-frame measured EEF delta with torque-level Cartesian impedance. In both
cases, both dexterous hands use six absolute physical motor targets. The A/B
experiments below select that one shared interface; they do not authorize
Revo2 to use JointTarget while Inspire uses Cartesian, or vice versa.

The locked from-scratch task pairs are now:

- `SimToolReal-Revo2-Franka-StaticOfficialJointTarget-Teacher-Direct-v0`;
- `SimToolReal-Revo2-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0`;
- `SimToolReal-Inspire-Franka-StaticOfficialJointTarget-Teacher-Direct-v0`;
- `SimToolReal-Inspire-Franka-StaticOfficialCartesianImpedance-Teacher-Direct-v0`.

The A/B decision is global: after both embodiments have been evaluated, one
arm interface is retained for both Revo2 and Inspire. Mixing JointTarget on one
hand with Cartesian impedance on the other is outside the accepted protocol.
Both candidates always command the six physical hand motors as absolute
targets; only the Franka arm coordinates differ.

Both use the same deterministic 40 x 40 x 80 mm, 30 g tabletop cube, the
Isaac Lab Franka home pose, six commanded Revo2 motors, five passive native
PhysX mimic joints, self-collision filters, force-contact diagnostics, strict
opposed grasp, 40 mm lift, and an 80 mm stable-hover target. Scripted reach,
close, lift, residual actions, post-success target locks, and action-coordinate
rewards are disabled. The only intended differences are the Franka action and
low-level controller: 7-D absolute joint targets versus 6-D measured
base-frame EEF delta with torque-level Cartesian impedance.

Open-hand runtime probes found:

| Diagnostic | Joint target | Cartesian impedance |
| --- | ---: | ---: |
| Policy action dimensions | 7 arm + 6 hand | 6 arm + 6 hand |
| Arm target error | 0.00474 rad max | <1 um position, 4.3e-6 rad rotation |
| OSC torque saturation | n/a | 0.0 |
| Minimum arm/table clearance | 3.92 mm | 4.47 mm |
| Robot contact with table/object/ground | none | none |
| Clearance violation/drop/termination | none | none |

The URDF-native distal mimic joints show a small damped open-hand transient.
For JointTarget, maximum passive-joint velocity fell from 0.038 rad/s at step
60 to 0.00135 rad/s at step 360. The 600-step probe then crossed the normal
8-second episode timeout and reset, so its later velocity spike and
`truncated=True` are not a physical instability. At step 300, the largest
passive distal position was 0.060 rad for JointTarget and 0.0054 rad for
Cartesian. There were no external contacts; this controller-pose-dependent
passive equilibrium is retained as an explicit observation rather than hidden
with an unreported mimic or damping change.

The first probes also exposed a 9 mm reward-baseline bug: the cube spawned at
z=0.345 m but physically settled at z=0.336 m, while lift progress retained the
spawn height. The task now spawns at the exact support center,
`table_top_z + cube_half_height = 0.296 + 0.040 = 0.336 m`. Runtime regression
measured `object_height_delta=5.96e-8 m` and `hover_z_error=0.07999995 m`.
Pre-fix probes remain useful for controller stability, but not for lift-height
or hover-progress comparisons.

The first 20-epoch JointTarget smoke run (`pn9wg6fc`) exposed another invalid
execution semantic. Per-step target rate limiting did not bound accumulated
target-to-measured error; the maximum reached 2.25 rad, while strict-grasp-seen
was 0.78%, force-grasp-seen was 1.10%, and success was 0%. This run is a
diagnostic failure, not an interface result. Protocol v2 clamps every executed
absolute arm target to the physical limits and to +/-0.04 rad around measured
joint position. A 60-step all-`+1` arm-action probe then measured 0.03725 rad
maximum tracking error with no external contact, clearance violation, drop,
or termination.

Matched protocol-v2 20-epoch smoke runs:

| Interface | W&B | Palm distance | Strict approach | Strict grasp seen | Force grasp seen | Success | Arm tracking error max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint absolute, measured bounded | `g55bha4q` | 0.1685 m | 0.4411 | 0% | 0% | 0% | 0.0424 rad |
| Base-frame Cartesian delta, torque impedance | `0rvzkb3f` | 0.2031 m | 0.2157 | 0% | 0% | 0% | 0.0120 rad |

Both runs used seed 142, 256 environments, horizon 32, and 155,648 frames.
This short smoke establishes finite/stable training and early reach shaping;
it is not an interface ranking. The paired 200-epoch runs must determine
whether opposed contact, force grasp, lift, and stable hover emerge.

### Joint-absolute formal-200 result

The corrected JointTarget run (`oajkc6qw`) completed 200 epochs from scratch.
Its best checkpoint was then evaluated deterministically for 256 balanced
episodes and recorded as a fixed-view 20-trial sequence (`9xmyzab8`).

| Metric | Result |
| --- | ---: |
| Strict task success | 0 / 256 (0.00%) |
| Strict true grasp seen | 2 / 256 (0.78%) |
| Force grasp seen | 210 / 256 (82.03%) |
| Force-stable grasp seen | 174 / 256 (67.97%) |
| Strict opposing contact seen | 116 / 256 (45.31%) |
| Strict thumb contact seen | 256 / 256 (100.00%) |
| Physical lift / stable hold | 0 / 256 (0.00%) |
| Maximum object-height change | 16.68 mm |

This is not a reachability or missing-contact failure. The fixed-view trace
shows the wrist reaching the cube, then continuing downward while the fingers
press from above or one side. The cube is usually tipped onto the table rather
than enclosed and lifted. Across the rendered 20 trials, mean peak object
height change was 7.50 mm while mean peak summed fingertip force was 86.1 N;
all 20 trials failed. The current single-stage reward therefore admits a
high-force contact local optimum that does not convert to opposed enclosure or
lift.

Authoritative artifacts:

- `logs/control_ablation/revo2_static_official_ab_v2/jointabsolute_bounded_fromscratch200_seed142`;
- `logs/debug_videos/static/revo2_official_jointabsolute_v2_ep200_best_256eval_20trial_20260715/20260715_211219/summary.json`;
- `logs/debug_videos/static/revo2_official_jointabsolute_v2_ep200_best_256eval_20trial_20260715/20260715_211219/videos/trial_sequence_000_trials_020_success_000_sr_0.000.mp4`.

### Cartesian formal-200 result

The matched torque-Cartesian run (`ath6y333`) also completed 200 epochs from
scratch. Checkpoint selection was based on physical acquisition rather than
mean return: epoch 175 had the strongest force-grasp screen and was evaluated
for 256 balanced episodes plus a fixed-view 20-trial sequence (`uwu9br2n`).

| Metric | Result |
| --- | ---: |
| Strict task success | 0 / 256 (0.00%) |
| Strict true grasp seen | 0 / 256 (0.00%) |
| Force grasp seen | 229 / 256 (89.45%) |
| Force-stable grasp seen | 223 / 256 (87.11%) |
| Strict opposing contact seen | 0 / 256 (0.00%) |
| Strict thumb contact seen | 1 / 256 (0.39%) |
| Physical lift / stable hold | 0 / 256 (0.00%) |
| Maximum object-height change | 5.56 mm |

The same local optimum therefore appears under both arm interfaces. Joint
absolute is closer to a geometrically opposed enclosure; Cartesian more often
produces a high-force contact without the thumb/opposition geometry. Neither
result supports choosing a controller yet. The shared cube objective, object
pose, and compact single-stage reward are the current bottleneck.

Authoritative Cartesian artifacts:

- `logs/control_ablation/revo2_static_official_ab_v2/cartesian_delta_impedance_fromscratch200_seed142`;
- `logs/debug_videos/static/revo2_official_cartesian_v2_ep175_256eval_20trial_20260715/20260715_214406/summary.json`;
- `logs/debug_videos/static/revo2_official_cartesian_v2_ep175_256eval_20trial_20260715/20260715_214406/videos/trial_sequence_000_trials_020_success_000_sr_0.000.mp4`.

### Archived 100% video audit

The earlier Revo2 JointTarget video with 20/20 trial success is a real
simulation lift, and its 256-episode evaluation reached 96.48%. It is not a
deployable action-interface result. Training telemetry for that checkpoint
shows up to 5.55 rad of arm target-to-measured error: the absolute target was
allowed to wind up far ahead of the robot and the position servo became an
unintended sustained-force controller. The checkpoint proves that the sphere
scene and state reward can discover lift behavior, but it cannot justify
keeping the unbounded action semantics.

Artifacts:

- `logs/rl_games/revo2_static_strict_jointtarget_nomimicfix_fromscratch_s142_512e400_20260715`;
- `logs/debug_videos/static/revo2_jointtarget_nomimicfix_ep400_strictfix_256eval_20trial_20260715/20260715_130217/summary.json`;
- `logs/debug_videos/static/revo2_jointtarget_nomimicfix_ep400_strictfix_256eval_20trial_20260715/20260715_130217/videos/trial_sequence_000_trials_020_success_020_sr_1.000.mp4`.

### Clean pre-lift stable-hover protocol

Protocol `static_stable_hover_from_scratch_control_ab_v7_clean_prelift`
removes the table-assisted scoop shortcut while keeping PPO fully from
scratch. There is no residual controller, scripted reach, scripted closure,
scripted lift, or grasp prior. Revo2 and Inspire use the same physical state
machine and reward contract:

- form a thumb-plus-two-finger opposed enclosure within 3 mm while the sphere
  remains within 5 mm of its settled tabletop height;
- maintain that enclosure, low object speed, low object-to-palm relative speed,
  and arm/hand table clearance for six consecutive control steps;
- terminate an episode if the sphere rises 12 mm before that clean-grasp latch;
- after the latch, require at least 4 cm of palm lift with at most 3 cm palm XY
  drift, bounded palm orientation drift, and bounded object-to-palm drift;
- finish in an 8 cm hover envelope for 18 low-speed control steps.

The privileged teacher additionally receives one shared 15-D semantic contact
state: five filtered fingertip-to-object force magnitudes, five strict
geometric fingertip contacts, opposed-contact state, object-to-palm slip
speed, table-clearance state, clean-grasp progress, and the clean-grasp latch.
These channels help phase recognition but cannot independently satisfy lift or
success. They are defined identically for Revo2 and Inspire and are reserved as
future student auxiliary targets.

The anti-scoop component probe passed for both embodiments: teleporting an
open-hand sphere to an 18 mm height delta produced termination on the same
control step. The old nominally 96.48%-successful Revo2 checkpoint scored
0/64 under the bounded clean-prelift task and produced no clean-grasp
candidate. This cross-task replay is diagnostic, not a new-policy success-rate
estimate, because the old checkpoint was trained under unbounded target
tracking and a different observation contract.

Runtime artifacts:

- `logs/debug_videos/component_tests/revo2_cleanhover_unclean_lift_gate_20260716/probe_summary_v2.json`;
- `logs/debug_videos/component_tests/inspire_cleanhover_unclean_lift_gate_20260716/probe_summary.json`;
- `outputs/probes/static_cleanhover_v7/revo_old_scoop_checkpoint_screen64/20260716_061534/summary.json`.

The first formal bounded Joint-absolute run used seed 142, 512 environments,
400 epochs, and online W&B run `r3yqdjo9`. Its independent deterministic
64-episode screen (`zvi4q7jz`) produced:

| Metric | Result |
| --- | ---: |
| Strict geometric grasp seen | 64 / 64 (100%) |
| Clean 3 mm candidate seen | 27 / 64 (42.19%) |
| Clean-grasp latch | 0 / 64 (0%) |
| Filtered multi-finger force grasp | 1 / 64 (1.56%; four frames) |
| Lift / stable hover / success | 0 / 64 (0%) |

This isolates a post-enclosure contact-settling bottleneck. The run learned a
repeatable thumb-plus-two-finger geometric enclosure from home, but did not
hold real opposed pressure with sufficiently low slip for six frames. Reward
audit also found that the inherited no-lift penalty and palm-lift reward became
active after strict geometric grasp, before the clean latch; they asked the
policy to move while the new protocol required it to settle.

Protocol `static_stable_hover_from_scratch_control_ab_v8_settle_then_lift`
removes that contradiction. Every lift incentive and no-lift penalty is gated
by the clean latch. Before the latch, a single 6k reward term promotes real
thumb/non-thumb pressure only when strict geometric opposition, table
clearance, and the original tabletop object height all hold. It retains the
3 mm, six-frame clean latch and all final lift/hover thresholds.

The v8 training run then exposed a second, independent reward problem. By
epoch 272 its episode return exceeded 1.23M while `success`, clean latch, and
hover goal all remained exactly zero. Current strict geometric grasp occupied
15.9% of sampled steps, so the policy was repeatedly farming acquisition and
hold shaping rather than completing the task. The run was intentionally
stopped; its machine-readable audit is
`logs/control_ablation/static_cleanhover_v8/reward_audit_summary.json`.

Protocol `static_stable_hover_from_scratch_control_ab_v9_task_reward_aligned`
keeps the environment and physical success definition unchanged and only
repairs credit assignment:

- reach/contact/opposition/underwrap shaping remains available before the
  clean latch and falls to 5% afterward;
- the independently farmable 6k strict-hold reward is disabled because the
  stricter 3 mm clean-candidate streak already represents pre-lift holding;
- the six-frame clean latch emits one 200k milestone bonus;
- strict 4 cm relative lift plus stable 8 cm hover emits one 1M terminal task
  bonus and still terminates the episode.

No curiosity model, residual, scripted action prior, target trajectory, or
demonstration is introduced. PPO remains from scratch and the new phase/event
scalars are logged independently from total return.

The v9 run showed that removing only strict-hold was not sufficient. At epoch
362 it reached 34.2% current strict grasp and a 2.55M episode return while the
clean latch and task success remained zero. The combined continuous approach,
touch, opposition, grasp-quality, and underwrap terms were still farmable as a
group. Its audit is
`logs/control_ablation/static_cleanhover_v9/reward_audit_summary.json`.

Protocol `static_stable_hover_from_scratch_control_ab_v10_progressive_acquisition`
therefore assigns each reward group one explicit lifetime. Coarse acquisition
shaping is fully active only until the first strict opposed grasp in an
episode. That event emits a one-time 100k milestone and permanently reduces
the coarse group to a 5% reacquisition floor. The existing clean-settle terms
remain active until the six-frame clean latch; lift/hover and the 1M terminal
task reward then take over. This is a reward-state transition only: policy
architecture, action interface, observations, reset, physics, and success
criteria remain unchanged.

### Global cross-hand static sphere protocol

Protocol `static_official_sphere_lift_global_action_ab_v1` combines the
previously learnable 44 mm sphere/state objective with the corrected official
six-motor hand models and the 0.04 rad measured-state arm-target bound. Revo2
and Inspire use the same Isaac Lab Franka home, sphere pose
`(0.58, -0.16, 0.320)`, reward, observation semantics, seed, and PPO budget.
The sphere settles to `z=0.318 m` before acquisition. The negative Y offset is
shared by both embodiments and keeps the sphere outside the longer Inspire
fingers' reset-settling sweep.

The candidate action contracts are global:

- Joint: 7-D bounded absolute Franka target + 6-D absolute hand-motor target;
- Cartesian: 6-D base-frame measured EEF delta through torque impedance +
  6-D absolute hand-motor target.

The final choice must be one row for both hands. A mixed Revo2-Joint and
Inspire-Cartesian result, or the reverse, is explicitly invalid.

Open-hand runtime parity at the final shared pose:

| Hand | Arm interface | Action dimensions | Object speed | External contacts | Drop | Max arm tracking error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Revo2 | Joint absolute | 7 + 6 | <1e-8 m/s | 0 | no | 0.00474 rad |
| Revo2 | Cartesian delta | 6 + 6 | <1e-8 m/s | 0 | no | 0.000304 rad |
| Inspire | Joint absolute | 7 + 6 | <1e-8 m/s | 0 | no | 0.02715 rad |
| Inspire | Cartesian delta | 6 + 6 | <1e-8 m/s | 0 | no | 0.00123 rad |

Runtime artifacts are under
`outputs/probes/static_official_sphere_global_action_ab_v1`.

Artifacts:

- `outputs/probes/revo2_static_official_ab_v1/joint_zero_open_600step_runtime.json`;
- `outputs/probes/revo2_static_official_ab_v1/cartesian_zero_open_300step_runtime.json`;
- `outputs/probes/revo2_static_official_ab_v1/joint_support_baseline_fix_runtime.json`;
- `outputs/probes/revo2_static_official_ab_v1/joint_absolute_measured_bound_saturated_runtime.json`.

## Locked v3 protocol

- Static 40 mm sphere at `(0.58, 0.0)` and deterministic home reset.
- Six real Revo2 motor commands in both policies.
- Joint arm: seven normalized absolute joint-position targets.
- Cartesian arm: world-frame `dx, dy, dz, droll, dpitch, dyaw` targets.
- Unit-Gaussian exploration is rate matched at `0.04 rad` maximum arm-joint
  target change per environment step. Unit-action probes measured maximum
  one-step palm motions of 0.68 mm / 0.0075 rad for joint and
  0.96 mm / 0.0051 rad for Cartesian.
- Reward terms that directly compare action coordinates or target changes are
  disabled. Lift progress and its phase observation use physical palm height
  for both interfaces.
- Scripted reach, close, lift, residual, and motion-planning priors are off.
- Stage 1 learns acquisition, Stage 2 sustained strict closure, and Stage 3
  strict object-coupled lift and stable hold.
- PPO uses seed 142, 512 environments, horizon 32, minibatch 16384, and five
  mini-epochs. Every run is from scratch or a same-interface continuation.

Runtime config parity audits for all three stages report exactly nine expected
differences: action contract, action and observation dimensions, interface and
reference names, plus the four Cartesian controller parameters. The audit
artifacts are under `outputs/probes/revo2_static_action_ab_v3_*cfg_parity*`.

## Invalidated pre-v3 evidence

The first joint Stage 1 had no arm target-rate limit, while PPO used fixed
`sigma=1`. A sampled joint action could therefore command nearly the full
Franka joint range, whereas Cartesian DLS clipped every joint correction to
0.04 rad. That run peaked at only 0.58% `strict_grasp_seen` and is not a valid
interface comparison.

After rate matching, the replacement joint run reached 1.32%
`strict_grasp_seen` by epoch 57 and 4.08% by epoch 133. The old run did not
reach 0.1% until epoch 195. This establishes that the earlier gap was primarily
an exploration-scale confound, not a broken Revo2 Cartesian controller or a
reward that only works for Cartesian actions.

The v2 run was stopped after this diagnostic because lift progress still used
joint projection for one interface and palm height for the other. v3 removes
that final state/reward mismatch before the definitive training runs.

## Results

### Stage 1: acquisition

| Interface | final grasp seen | peak grasp seen | final current strict grasp | peak current strict grasp |
| --- | ---: | ---: | ---: | ---: |
| Joint target | 78.89% | 85.94% | 2.38% | 4.67% |
| Cartesian delta | 37.73% | 59.95% | 0.53% | 2.19% |

Both policies learn physical thumb/non-thumb opposition from scratch. After
physical exploration scaling is matched, joint targets are faster for this
seed and static reset. Neither Stage 1 policy is expected to hold a grasp;
current strict grasp is deliberately addressed by Stage 2.

W&B runs:

- Joint: `1rloc7fe`
- Cartesian: `elod35lc`

Stage 2, Stage 3, strict evaluation, and 20-trial video results will be added
as they complete.

## v3 postmortem: absolute-target windup

The Stage-1 acquisition numbers above are real contact statistics, but they do
not establish a deployable joint controller. Fixed-view traces exposed a
second action-semantic mismatch:

- The v3 "Joint target" policy outputs a target relative to the home pose.
  Repeated saturated actions walk the rate-limited target toward a joint limit.
- The Cartesian policy solves a local correction from the measured wrist pose.
- IsaacGym's Revo2 execution path instead uses incremental joint-position
  targets: `target_t = target_(t-1) + speed_scale * dt * action_t`.

In the corrected-geometry v6 Joint run, the learned palm/object state looked
correct in aggregate, but the deterministic trace showed up to 0.36 rad of
arm target-tracking error. Replaying the identical arm commands while keeping
the hand open changed the wrist trajectory substantially. The original policy
was using finger/table contact to support a lagging, saturated arm command.
Its Stage-2 continuation therefore reached 0.81 mean strict non-thumb contacts
but no strict opposing grasp or lift after 300 epochs. This is a controller
artifact, not evidence that joint-position control is intrinsically unable to
grasp.

## Corrected v4 protocol

The definitive comparison retains the archived absolute-target tasks and adds
three `JointDelta` tasks. Their 13-D policy action is:

- seven incremental Franka joint-position commands integrated from the
  previous target;
- six real Revo2 motor commands with the same semantic mapping as Cartesian;
- two controller applications per policy step (`decimation=2`) and a maximum
  0.04 rad cumulative arm-target correction per policy step. A matched runtime
  probe measured 0.040 rad for JointDelta and 0.047 rad for the Cartesian DLS
  controller under saturated compound commands;
- physical joint-limit clamping before targets are sent to PhysX.

All reset, observation, reward, curriculum, seed, PPO, and hand-control fields
remain shared. The v4 Stage-1, Stage-2, and Stage-3 runs must replace v3 as the
deployable Joint-versus-Cartesian conclusion.

## v7 runtime audit and corrected v8 bound

The first `JointDelta` run exposed one more runtime confound. Limiting each
increment does not limit the accumulated target when integration starts from
the previous target. Its epoch-400 deterministic trace reached the calibrated
object/palm pose but accumulated up to 1.67 rad of target-to-measured arm error;
the policy was again using a lagging position target as an unintended force
command. It produced 52.9% mean strict non-thumb contact and zero thumb contact.

The corrected controller retains incremental joint-space actions but clamps
every arm target to both the physical joint limits and a 0.04 rad envelope
around measured joint position. This is the joint-space analogue of the
Cartesian DLS controller, whose runtime probe had 0.037 rad target-tracking
error. The environment now logs mean and maximum arm target-tracking error on
every PPO epoch. The v8 from-scratch run, not v7, is the definitive JointDelta
baseline.

## v8 reward audit and v9 single-variable correction

With target tracking bounded, v8 briefly reached 0.70% `strict_grasp_seen`,
then converged to the calibrated palm pose with 52.5% non-thumb contact but
zero thumb contact. Its deterministic trace stayed within 0.0393 rad of
measured arm state, while both thumb action channels converged to fully open.
This separates the remaining failure from controller semantics: the 12,000
weight palm-frame reach term was larger than the complete Stage-1 opposition
objective and rewarded preserving pose instead of closing the thumb.

The v9 protocol changes exactly one shared state-reward coefficient:
`dynamic_tabletop_palm_frame_pregrasp_rew_scale` is reduced from 12,000 to
1,800. The calibrated target remains unchanged and all controller, reset,
observation, contact, seed, and PPO fields remain locked. This makes the pose
term a reach guide while thumb/non-thumb opposition determines acquisition.

## 2026-07-16 distal-force hold audit

The official-sphere v5/v6 continuations tested whether a Stage-1 acquisition
policy could be converted into a sustained distal thumb/non-thumb force pair.
A new vector trace recorded all five filtered distal-to-object forces, the
force streak, actions, object/palm relative velocity, object height, and
object-to-palm displacement for every environment and control step.

The Stage-1 checkpoint produced a distal force pair in 39/64 environments but
for at most two consecutive frames. The v5 force-only continuation reduced
coverage to 9/64 and learned sustained non-thumb pressure with almost no thumb
contact. The v6 opposed-pressure continuation briefly reached 41/64 coverage
and a six-frame maximum streak at epoch 15, then regressed again. There were no
one-frame sensor gaps, so debounce was not the issue.

This route is stopped as an objective, not merely retuned. Auditing the
archived 96.48% physical-lift policy showed that the sphere stayed within a
few millimeters of a fixed palm-frame pose while rising about 0.23 m, even
though the five distal sensors reported no force grasp during the carry. The
load was transferred through the palm and proximal finger links. Requiring a
distal thumb/non-thumb pair therefore rejected a physically valid grasp and
drove PPO away from useful enclosure behavior.

Subsequent controller selection uses one global action family for Revo2 and
Inspire and accepts policies by physical lift plus stable palm/object coupling
and hover, with distal forces retained only as contact diagnostics. The v5/v6
artifacts remain under
`outputs/probes/static_official_sphere_global_action_ab_v5` and
`outputs/probes/static_official_sphere_global_action_ab_v6`.

## v9 Stage 2 hold and velocity bridge

The first v9 Stage-2 continuation learned a genuine stationary enclosure:
78.3% current strict grasp, a 93-step mean strict streak, and 0.55 mean
underwrap score. A deterministic trace held strict contact for 239 consecutive
frames with target tracking bounded to 0.041 rad. It produced exactly 0 mm of
palm lift, however, so the remaining failure was vertical-motion exploration,
not grasp acquisition or load-bearing geometry.

The follow-up keeps the same Stage-2 checkpoint and adds two state-only terms
already implemented by the environment: simultaneous positive palm/object Z
velocity, and the same velocity gated by low lateral and angular palm motion.
The bridge is shared by JointDelta and Cartesian Stage 2. It supplies no joint
direction, IK target, action prior, residual, or scripted trajectory.

The velocity-only continuation still converged to a stationary downward press:
88.8% current strict grasp, roughly 14 mm negative palm displacement, and no
object lift. The audit found that the 40,000-scale strict-hold term continued
paying its maximum value forever after the 20-step hold milestone. The next
continuation stops only that term once 20 consecutive strict steps are
complete. Grasp-loss penalties and all physical lift/carry rewards remain
active, so releasing the object is never made profitable.

## v12 privileged-state encoder A/B

The static teacher does not receive a point cloud. Its privileged observation
already contains the complete state needed for a fixed sphere, but the default
rl_games MLP mixes all channels from its first layer. To test whether this
representation is causing sample inefficiency, v12 compares two actor-critic
networks without changing the 95-D observation, reward, PPO settings, reset,
controller, or random seed:

- `flat`: the existing shared `[512, 512, 256, 128]` MLP;
- `structured`: four one-layer semantic encoders followed by a shared
  `[512, 256, 128]` fusion MLP.

The structured branches are robot state plus previous action (39-D), object
pose and velocity (13-D), palm/fingertip relative geometry (24-D), and
contact/task phase (19-D). Each branch maps to 144 dimensions. The Joint
networks have approximately 478k and 476k parameters respectively, a
difference below one percent. Input normalization remains enabled in both.

The comparison is tracked under W&B group `static_state_encoder_ab_v12` and
each local run writes `resolved_run_config.json`. Primary diagnostics are the
first-strict-grasp milestone rate, current strict opposition, clean-latch
rate, and strict stable-hover success. A single-seed improvement is treated as
evidence of better sample efficiency, not yet as a final architecture choice;
the winner must repeat across seeds before being shared by Revo2 and Inspire.

Point-cloud encoding is a separate later ablation for multi-object and
affordance-conditioned teachers. Adding a clean mesh point cloud to this
single-sphere control test would change both information and architecture and
would therefore obscure the current diagnosis.

### v12 encoder result

The seed-42, 512-environment runs completed 400 epochs:

| Encoder | parameters | final return | peak strict seen | peak strict true grasp | peak clean candidate | peak clean latch | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat | 477,851 | 2,140,791 | 49.55% | 1.11% | 0.0061% | 0% | 0% |
| Four-branch structured | 475,723 | 132,598 | 0% | 0% | 0% | 0% | 0% |
| Flat + per-finger relation residual | 491,547 | 3,183,624 | 38.86% | 1.19% | 0.0183% | 0% | 0% |

The flat policy separately drove strict thumb contact to 89.48%, mean strict
non-thumb contact count to 0.921, and strict opposing contact to 88.24%, but
almost never satisfied them simultaneously. This is a contact-coordination
failure, not an absent object-state signal. The structured policy learned the
smooth pregrasp geometry faster early in training, then stayed outside strict
contact for the full run. Coarse semantic branching is therefore rejected.
The third run preserves the entire flat path and adds a shared encoder over
five paired tokens containing fingertip-relative xyz, surface distance,
contact force, and contact flag. Its zero-initialized residual head makes its
initial policy exactly equal to the same-seed flat policy. It increased strict
true grasp slightly and produced three times as many clean candidates, but
still never maintained one for the required six frames.

W&B runs:

- Flat: `mo1qlf15`
- Structured: `azvb9o50`
- Flat plus finger relation residual: `1zv9kgft`

The machine-readable comparison is
`logs/control_ablation/static_state_encoder_v12/comparison_summary.json`.
Because this is one seed, it does not establish a universal architecture
ranking. It does establish that coarse block encoding is not the missing
ingredient, while explicit finger-object relations provide only a modest
coordination gain. The remaining discontinuity is between the frequently
reached 8 mm strict-contact event and the 3 mm, low-relative-speed, six-frame
clean latch. v13 therefore keeps the relation residual but curricula only
those clean-contact thresholds during training, then forces alpha 1.0 for the
unchanged strict evaluation contract.

## 2026-07-16 mature-baseline reconstruction and canonical v2.5

The earlier static experiments were harder to interpret than necessary
because they combined a new home-pose tabletop task, a three-second stable
hover requirement, hand-physics corrections, and several experimental reward
gates. They were not a faithful control reconstruction of the mature
SimToolReal/Play2Perfect training topology.

The local IsaacGym implementations establish the following useful baseline:

- Franka actions are incremental targets accumulated from the previous target;
- hand actions are absolute joint targets;
- approach reward is best-so-far progress rather than a persistent distance
  reward;
- object lift reward stops at the lift threshold;
- after lift, goal-keypoint progress and goal reach replace continued upward
  motion as the main objective;
- the default object/reset distribution is substantially easier than the new
  upright-home tabletop benchmark, and the published-style runs use much
  larger environment and transition budgets.

Protocol `static_canonical_progress_lift_target_v2_5` reconstructs that
topology while preserving the deployment contract. Both hands use seven
bounded Franka joint-target increments and six absolute physical hand-motor
targets. There is no residual policy, scripted reach, scripted close,
scripted lift, demonstration, or motion-planning prior. The reset curriculum
uses a mixed distribution with 20% object-aligned pregrasp resets, 20% current
hard resets, and 60% interpolation between them; the hard endpoint is the
official upright Franka home pose. Reward switches from best-so-far approach
and lift progress to a fixed 10 cm hover target once the lift threshold is
reached.

This reconstruction exposed two concrete earlier failures:

1. Protocol v22 kept paying persistent object-height reward after the target
   height. A Revo2 epoch-450 policy achieved 98.4% strict grasp and 89.5%
   physical lift but zero stable-target success because it threw the object to
   approximately 0.91 m. This was a reward-stage bug, not failed grasp
   exploration.
2. A synchronous reset curriculum moved every environment from pregrasp to
   home at the same time and caused catastrophic forgetting. The v2.5 mixed
   distribution keeps solved acquisition states in the replay distribution
   while progressively adding harder starts.

Deterministic strict evaluation results currently available at the pregrasp
endpoint (`canonical-reset-alpha=0`) are:

| Hand/checkpoint | episodes | strict grasp | physical lift | stable hold | 3 s task success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inspire ep150 | 256 | 100% | 100% | 89.45% | 16.02% |
| Inspire ep200 | 256 | 100% | 100% | 100% | 75.00% |
| Inspire ep250 | 256 | 100% | 100% | 99.61% | 86.33% |
| Revo2 ep450 | 256 | 100% | 98.05% | 76.56% | 4.69% |

The Inspire ep250 fixed-camera sequence independently produced 17/20 strict
successes. It is an RL rollout from the same checkpoint, not a scripted
demonstration. The Revo2 ep450 checkpoint reached 25.78% when only the hold
duration was diagnostically reduced from 180 to 18 steps, so its remaining
failure is real hover oscillation rather than only an overly long acceptance
window. `post_clean_grip_retained` is a failure-funnel diagnostic and is not a
success gate; final success is the consecutive intersection of physical
grasp, lift height, object/palm relative stability, fixed hover pose, wrist
drift bounds, and arm/table clearance.

The Inspire epoch-700 checkpoint was subsequently evaluated at the official
upright Franka home endpoint (`canonical-reset-alpha=1`). It passed 256/256
deterministic episodes (100%). Every episode reached strict grasp, physical
lift, stable hover, and the full 180-step success streak. The authoritative
seed-42 summary is
`outputs/eval_rl_games/static_canonical_v25_inspire_ep700_home_256eval_20260716/20260716_170947/summary.json`.
A second seed-142 evaluation also passed 256/256, for 512/512 total; its
summary is
`outputs/eval_rl_games/static_canonical_v25_inspire_ep700_home_seed142_256eval_20260716/20260716_172441/summary.json`.
This is the first accepted vector result for the canonical static benchmark;
its single-environment fixed-view video passed 20/20 trials and is curated at
`outputs/curated_videos/teacher/static/inspire_static_home_jointdelta_v25_ep700_20trial_sr1.000.mp4`.

Revo2 remains unaccepted until its `canonical-reset-alpha=1` evaluation
reaches at least 80% over 256 episodes and is reproduced in a fixed-view
20-trial video. No JointDelta-versus-Cartesian conclusion is drawn from v2.5
yet; the matched Cartesian run is evaluated independently while Revo2
JointDelta robustness is repaired.

### Revo2 home robustness audit and v2.6

The completed Revo2 v2.5 JointDelta run did not meet acceptance. Strict
seed-42 home evaluations were 88/256 (34.38%) at epoch 950, 102/256 (39.84%)
at epoch 1400, and 90/256 (35.16%) at epoch 1600. Epoch 1400 grasped in
255/256 episodes and lifted in 254/256, so acquisition is solved in the large
vector evaluation; the remaining loss is the continuous three-second hover.

The fixed-camera single-environment epoch-1400 sequence was materially worse:
2/20 strict successes. It is retained, including config and trace, at
`logs/debug_videos/static/revo2_jointdelta_v25_ep1400_home_20trial_20260716/20260716_175055/videos/trial_sequence_000_trials_020_success_002_sr_0.100.mp4`.
Its trace contains both acquisition misses and lifted grasps whose object
angular velocity does not remain below 3 rad/s. With reset noise disabled,
one exact-home environment grasped and lifted but did not complete the hold;
16 numerically identical replicated environments produced 4/16 successes.
This establishes contact-dynamics sensitivity rather than an observation
origin leak. An evaluator-only counterfactual that zeroed Franka actions after
the hover latch reduced success to 0/256, so hard arm freezing is not a valid
stability fix.

A non-rendered single-environment evaluation over 64 consecutive resets also
produced 0/64 successes (42.19% strict grasp, 35.94% lift, 9.38% ever-stable),
confirming that the large vector evaluation is not a sufficient deployment
metric. Future acceptance therefore requires both balanced vector evaluation
and repeated single-environment evaluation/video. The matched Cartesian v2.5
run was 0/256 at epoch 1000, then improved to 65/256 (25.39%) at epoch 1200;
it learns later than JointDelta but had not surpassed the 102/256 JointDelta
best at that point.

Protocol `static_canonical_robust_home_stable_v2_6` preserves v2.5 and changes
only shared training parameters: 10% pregrasp anchors, 50% full-home anchors,
0.02 rad reset noise, a 0.015 rad JointDelta step, and stronger existing
hover/stability/hold rewards. It adds no reward source, script, residual, or
planner. A fresh Revo2 run is online at
`https://wandb.ai/simonlsx/simtoolreal_lab/runs/xql0f5hc`.

### 2026-07-17 action decision and strict single-environment audit

The matched v2.5 Cartesian run eventually learned the task in the replicated
vector setting, but did not reproduce robustly in one physical scene. The
v2.6 JointDelta run was trained from random weights and is materially stronger
under every deployment-oriented check:

| Interface/checkpoint | 256-env seed 42 | 256-env seed 142 | 1-env 64 trials | Fixed-view 20 trials |
| --- | ---: | ---: | ---: | ---: |
| Cartesian v2.5 ep1400 | 119/256 (46.48%) | not run | 7/64 (10.94%) | 3/20 (15.00%) |
| JointDelta v2.6 ep1600, 10 s | 219/256 (85.55%) | 222/256 (86.72%) | 41/64 (64.06%) | 14/20 (70.00%) |
| JointDelta v2.6 ep1600, 12 s evaluation budget | unchanged policy | unchanged policy | 47/64 (73.44%) | 17/20 (85.00%) |
| JointDelta v2.6 ep1600, 14 s evaluation budget | unchanged policy | unchanged policy | 54/64 (84.38%) | use accepted 12 s video above |

The 12- and 14-second diagnostics change only the episode time budget. They do
not change policy weights, actions, reward, reset state, success geometry, or
the required 180-step continuous stable-hover streak. At 14 seconds the strict
single-environment result passes the 80% acceptance threshold with 54/64
successes. Strict grasp, physical lift, and stable-hold episode rates are
98.44%, 95.31%, and 93.75%, respectively. The ten failures are classified as
five post-acquisition grip-retention failures, three short success streaks,
one missing lift, and one missing strict thumb contact.

A per-environment origin diagnostic found no monotonic dependence on distance
from the replicated grid origin. The remaining vector-versus-single-scene gap
is therefore treated as contact-rich PhysX batching sensitivity, not as an
unsubtracted world-frame observation. The 14-second repeated single-scene
result now independently clears the threshold, so acceptance no longer relies
on the stronger replicated vector score.

The arm action choice itself is now unambiguous: Cartesian is rejected and the
global action is a bounded seven-dimensional Franka JointDelta plus six
absolute physical hand-motor targets. Revo2 static is accepted. A brief v2.7
probe with a 60-scale state-only hover-overshoot penalty was rejected after its
training return collapsed to -136k at epoch 400; its task branch and local run
were removed rather than carried into the main codebase.

Inspire was then trained from random weights under exactly the same v2.6 action
contract, static task, reset curriculum, PPO settings, physical success test,
and fixed camera. The selected epoch-450 checkpoint achieves 256/256 for seed
42, 256/256 for seed 142, 64/64 repeated trials in one physical environment,
and 20/20 in the continuous fixed-view video. This closes the static phase:
both hands pass the deployment-oriented 80% threshold with one shared action
interface. The Inspire online W&B run is
`https://wandb.ai/simonlsx/simtoolreal_lab/runs/1uxwe0d4`.

Authoritative artifacts:

- `outputs/eval_rl_games/static_canonical_v26_revo2_joint_ep1600_home_256eval_20260717/20260717_070630/summary.json`;
- `outputs/eval_rl_games/static_canonical_v26_revo2_joint_ep1600_home_seed142_256eval_20260717/20260717_070809/summary.json`;
- `outputs/eval_rl_games/static_canonical_v26_revo2_joint_ep1600_home_12s_singleenv_64eval_20260717/20260717_072057/summary.json`;
- `outputs/eval_rl_games/static_canonical_v26_revo2_joint_ep1600_home_14s_singleenv_64eval_20260717/20260717_074800/summary.json`;
- `logs/debug_videos/static/revo2_jointdelta_v26_ep1600_home_12s_20trial_20260717/20260717_072046/videos/trial_sequence_000_trials_020_success_017_sr_0.850.mp4`;
- `logs/debug_videos/static/revo2_cartesian_v25_ep1400_home_20trial_20260716/20260716_184111/videos/trial_sequence_000_trials_020_success_003_sr_0.150.mp4`;
- `outputs/eval_rl_games/static_canonical_v26_inspire_joint_ep450_home_256eval_20260717/20260717_080955/summary.json`;
- `outputs/eval_rl_games/static_canonical_v26_inspire_joint_ep450_home_seed142_256eval_20260717/20260717_081603/summary.json`;
- `outputs/eval_rl_games/static_canonical_v26_inspire_joint_ep450_home_singleenv_64eval_20260717/20260717_081149/summary.json`;
- `logs/debug_videos/static/inspire_jointdelta_v26_ep450_home_20trial_20260717/20260717_081157/videos/trial_sequence_000_trials_020_success_020_sr_1.000.mp4`.

### Rolling transfer audit and v2.7

The accepted static action was transferred without changing coordinates or
authority: seven bounded Franka JointDelta actions plus six absolute physical
hand-motor targets. The first rolling transfer, protocol
`rolling_multishape_jointdelta_v2_6`, exposed a reward failure rather than an
action failure. Its inherited rolling objective paid large persistent
geometric contact and opposition rewards. At epoch 700, both policies still
scored 0/256 on the easiest static-sphere pregrasp evaluation:

| Hand | geometric strict grasp | force grasp | physical lift | success |
| --- | ---: | ---: | ---: | ---: |
| Revo2 | 250/256 (97.66%) | 0/256 | 0/256 | 0/256 |
| Inspire | 10/256 (3.91%) | 15/256 (5.86%) | 0/256 | 0/256 |

The Revo2 policy is the clearest counterexample: its training return exceeded
13 million while it learned to geometrically surround the sphere without
forming a multi-finger force closure or lifting. The v2.6 runs were stopped at
this audit instead of being reported as progress. Authoritative summaries are
`outputs/eval_rl_games/rolling_v26_revo2_ep700_static_pregrasp_256eval_20260717/20260717_091458/summary.json`
and
`outputs/eval_rl_games/rolling_v26_inspire_ep700_static_pregrasp_256eval_20260717/20260717_091501/summary.json`.

Protocol `rolling_multishape_jointdelta_v2_7_lift_first` keeps the action,
robot models, PPO network, and strict force-backed success definition fixed.
It changes the curriculum/reward contract only:

1. acquisition uses the accepted canonical best-so-far approach progress;
2. persistent geometric contact reward is zero;
3. sustained return comes from object lift, hover convergence, and stable hold;
4. pregrasp-to-home reset difficulty is gated by physical catch-hold EMA;
5. 0.1-0.4 m/s speed and additional assets stay locked until reset alpha is
   at least 0.98.

Both v2.7 policies start from random weights and run online at:

- Revo2: `https://wandb.ai/simonlsx/simtoolreal_lab/runs/du8vh85i`;
- Inspire: `https://wandb.ai/simonlsx/simtoolreal_lab/runs/slbz99z0`.

Reset/physics smoke videos and their config/metrics sidecars are stored at
`logs/debug_videos/rolling/revo2_jointdelta_v27_liftfirst_reset_smoke_20260717/video.mp4`
and
`logs/debug_videos/rolling/inspire_jointdelta_v27_liftfirst_reset_smoke_20260717/video.mp4`.
