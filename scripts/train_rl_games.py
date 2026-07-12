#!/usr/bin/env python3
"""Train SimToolReal IsaacLab tasks with rl_games."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

EXT_SOURCE = Path(__file__).resolve().parents[1] / "source" / "simtoolreal_lab"
if str(EXT_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXT_SOURCE))
REPO_ROOT = Path(__file__).resolve().parents[1]

from isaaclab.app import AppLauncher


def _largest_divisor_at_most(value: int, upper_bound: int) -> int:
    upper_bound = max(1, min(value, upper_bound))
    for candidate in range(upper_bound, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="SimToolReal-Revo2-Franka-StaticBall-Grasp-Direct-v0",
    help="Gym task id.",
)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=512, help="Number of envs.")
parser.add_argument("--seed", type=int, default=None, help="Training seed. Uses the YAML default when omitted.")
parser.add_argument("--max-epochs", type=int, default=None, help="Override rl_games max_epochs.")
parser.add_argument("--horizon-length", type=int, default=None, help="Override rl_games horizon_length.")
parser.add_argument("--minibatch-size", type=int, default=None, help="Override rl_games minibatch_size.")
parser.add_argument("--mini-epochs", type=int, default=None, help="Override rl_games mini_epochs.")
parser.add_argument("--save-frequency", type=int, default=None, help="Override rl_games save_frequency.")
parser.add_argument("--score-to-win", type=float, default=None, help="Override rl_games score_to_win.")
parser.add_argument("--learning-rate", type=float, default=None, help="Override rl_games learning_rate.")
parser.add_argument(
    "--lr-schedule",
    choices=("adaptive", "linear", "identity", "fixed"),
    default=None,
    help="Override rl_games lr_schedule. Use fixed/identity to keep --learning-rate constant.",
)
parser.add_argument("--kl-threshold", type=float, default=None, help="Override rl_games adaptive KL threshold.")
parser.add_argument(
    "--log-root",
    default=str(REPO_ROOT / "logs" / "rl_games"),
    help="Root directory for rl_games runs.",
)
parser.add_argument("--experiment-name", default=None, help="Run directory name under --log-root.")
parser.add_argument("--checkpoint", default="", help="Optional checkpoint to restore.")
parser.add_argument(
    "--reset-optimizer-on-load",
    action="store_true",
    help="Load model/env state from --checkpoint but clear optimizer momentum and apply the configured learning rate.",
)
parser.add_argument(
    "--reset-epoch-on-load",
    action="store_true",
    help="Reset checkpoint epoch/frame/best-reward counters after loading weights.",
)
parser.add_argument("--sigma", type=float, default=None, help="Optional fixed policy sigma override.")
parser.add_argument(
    "--dynamic-curriculum-alpha",
    type=float,
    default=None,
    help="Override dynamic speed curriculum alpha during training. Useful for fixed-difficulty fine-tuning.",
)
parser.add_argument(
    "--tabletop-asset-curriculum-alpha",
    type=float,
    default=None,
    help="Override tabletop asset curriculum alpha during training.",
)
parser.add_argument(
    "--tabletop-motion-curriculum-alpha",
    type=float,
    default=None,
    help="Override tabletop motion-mode curriculum alpha during training.",
)
parser.add_argument(
    "--tabletop-pregrasp-lead-time",
    type=float,
    default=None,
    help="Override the moving-object velocity lead time during training.",
)
parser.add_argument(
    "--tabletop-pregrasp-ahead-distance",
    type=float,
    default=None,
    help="Override the fixed moving-object interception lead distance during training.",
)
parser.add_argument(
    "--scripted-action-prior-active-residual-scale",
    type=float,
    default=None,
    help="Override policy residual authority while a scripted action prior is active.",
)
parser.add_argument(
    "--scripted-relative-lift-target-scale",
    type=float,
    default=None,
    help="Scale the configured relative joint-space lift target delta.",
)
parser.add_argument(
    "--scripted-action-prior-lift-steps",
    type=int,
    default=None,
    help="Override the scripted lift duration in control steps.",
)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    help="Override the environment episode duration in seconds.",
)
parser.add_argument(
    "--dynamic-success-hold-steps",
    type=int,
    default=None,
    help="Override consecutive stable steps required before success latches.",
)
parser.add_argument(
    "--tabletop-post-success-hand-close-fraction",
    type=float,
    default=None,
    help="Override the semantic 6-DoF hand-close blend applied after success latches.",
)
parser.add_argument(
    "--stability-target-latch-min-success-streak",
    type=int,
    default=None,
    help=(
        "Optionally lock configured stability targets before final success after this many "
        "consecutive strict stable steps. The final success hold requirement is unchanged."
    ),
)
parser.add_argument(
    "--freeze-input-running-stats",
    action="store_true",
    help=(
        "Keep loaded observation running mean/std fixed during continuation training. "
        "This is useful when fine-tuning a working checkpoint with a lightly changed reward."
    ),
)
parser.add_argument(
    "--freeze-value-running-stats",
    action="store_true",
    help="Keep loaded value/reward running mean/std fixed during continuation training.",
)
parser.add_argument(
    "--behavior-anchor-checkpoint",
    default="",
    help=(
        "Optional frozen reference policy checkpoint for behavior/KL anchored continuation. "
        "When omitted and --behavior-anchor-coef > 0, --checkpoint is used."
    ),
)
parser.add_argument(
    "--behavior-anchor-coef",
    type=float,
    default=0.0,
    help="Weight for the frozen reference-policy behavior anchor added to PPO loss.",
)
parser.add_argument(
    "--behavior-anchor-mode",
    choices=("mu_mse", "kl"),
    default="mu_mse",
    help="Anchor loss type: action-mean MSE or Gaussian policy KL(current || reference).",
)
parser.add_argument(
    "--behavior-anchor-hand-start-index",
    type=int,
    default=None,
    help=(
        "Optional first hand-action index. When set, arm/hand weights below are applied; "
        "for Franka+Revo2 this is normally 7."
    ),
)
parser.add_argument(
    "--behavior-anchor-arm-weight",
    type=float,
    default=1.0,
    help="Per-action anchor weight for indices before --behavior-anchor-hand-start-index.",
)
parser.add_argument(
    "--behavior-anchor-hand-weight",
    type=float,
    default=1.0,
    help="Per-action anchor weight for indices from --behavior-anchor-hand-start-index onward.",
)
parser.add_argument("--rl-device", default=None, help="Device for the policy. Defaults to --device.")
parser.add_argument("--wandb-project", default=None, help="Enable W&B logging under this project.")
parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name. Defaults to the rl-games run name.")
parser.add_argument("--wandb-group", default=None, help="Optional W&B group.")
parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
parser.add_argument(
    "--wandb-mode",
    choices=("online", "offline", "disabled"),
    default=os.environ.get("WANDB_MODE", "online"),
    help="W&B mode. Defaults to WANDB_MODE or online.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from rl_games.algos_torch import torch_ext  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import simtoolreal_lab  # noqa: F401,E402


def _trace(message: str) -> None:
    print(f"[TRAIN] {message}", flush=True)


def _maybe_init_wandb(args: argparse.Namespace, agent_cfg: dict, run_dir: Path, experiment_name: str):
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested, but wandb is not installed in this environment.") from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    config = agent_cfg["params"]["config"]
    wandb_init_timeout = float(os.environ.get("WANDB_INIT_TIMEOUT", "120"))
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or experiment_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(run_dir),
        sync_tensorboard=True,
        job_type="train",
        settings=wandb.Settings(init_timeout=wandb_init_timeout),
        config={
            "task": args.task,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": args.device,
            "rl_device": args.rl_device or args.device,
            "checkpoint": args.checkpoint,
            "reset_optimizer_on_load": args.reset_optimizer_on_load,
            "reset_epoch_on_load": args.reset_epoch_on_load,
            "freeze_input_running_stats": args.freeze_input_running_stats,
            "freeze_value_running_stats": args.freeze_value_running_stats,
            "dynamic_curriculum_alpha": args.dynamic_curriculum_alpha,
            "tabletop_asset_curriculum_alpha": args.tabletop_asset_curriculum_alpha,
            "tabletop_motion_curriculum_alpha": args.tabletop_motion_curriculum_alpha,
            "tabletop_pregrasp_lead_time": args.tabletop_pregrasp_lead_time,
            "tabletop_pregrasp_ahead_distance": args.tabletop_pregrasp_ahead_distance,
            "scripted_action_prior_active_residual_scale": args.scripted_action_prior_active_residual_scale,
            "scripted_relative_lift_target_scale": args.scripted_relative_lift_target_scale,
            "scripted_action_prior_lift_steps": args.scripted_action_prior_lift_steps,
            "episode_length_s": args.episode_length_s,
            "dynamic_success_hold_steps": args.dynamic_success_hold_steps,
            "tabletop_post_success_hand_close_fraction": (
                args.tabletop_post_success_hand_close_fraction
            ),
            "stability_target_latch_min_success_streak": (
                args.stability_target_latch_min_success_streak
            ),
            "behavior_anchor_checkpoint": args.behavior_anchor_checkpoint,
            "behavior_anchor_coef": args.behavior_anchor_coef,
            "behavior_anchor_mode": args.behavior_anchor_mode,
            "behavior_anchor_hand_start_index": args.behavior_anchor_hand_start_index,
            "behavior_anchor_arm_weight": args.behavior_anchor_arm_weight,
            "behavior_anchor_hand_weight": args.behavior_anchor_hand_weight,
            "agent_config": agent_cfg,
            "max_epochs": config.get("max_epochs"),
            "horizon_length": config.get("horizon_length"),
            "minibatch_size": config.get("minibatch_size"),
            "mini_epochs": config.get("mini_epochs"),
            "learning_rate": config.get("learning_rate"),
            "lr_schedule": config.get("lr_schedule"),
            "kl_threshold": config.get("kl_threshold"),
        },
    )


def _configure_agent(agent_cfg: dict, env, args: argparse.Namespace) -> tuple[dict, Path, str]:
    agent_cfg = copy.deepcopy(agent_cfg)
    params = agent_cfg["params"]
    config = params["config"]

    rl_device = args.rl_device or args.device
    config["device"] = rl_device
    config["device_name"] = rl_device
    config["num_actors"] = env.num_envs

    if args.seed is not None:
        params["seed"] = args.seed

    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        config["horizon_length"] = args.horizon_length
    if args.mini_epochs is not None:
        config["mini_epochs"] = args.mini_epochs
    if args.save_frequency is not None:
        config["save_frequency"] = args.save_frequency
    if args.score_to_win is not None:
        config["score_to_win"] = args.score_to_win
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.lr_schedule is not None:
        config["lr_schedule"] = "identity" if args.lr_schedule == "fixed" else args.lr_schedule
    if args.kl_threshold is not None:
        config["kl_threshold"] = args.kl_threshold

    train_dir = Path(args.log_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_token = args.task.removeprefix("SimToolReal-").removesuffix("-Direct-v0").lower().replace("-", "_")
    experiment_name = args.experiment_name or (
        f"{config['name']}_{task_token}_{env.num_envs}env_{config['max_epochs']}ep_{timestamp}"
    )
    config["train_dir"] = str(train_dir)
    config["full_experiment_name"] = experiment_name

    batch_size = int(config["horizon_length"]) * int(env.num_envs)
    requested_minibatch = int(args.minibatch_size or config["minibatch_size"])
    config["minibatch_size"] = _largest_divisor_at_most(batch_size, requested_minibatch)

    max_epochs = int(config["max_epochs"])
    if max_epochs > 0:
        config["save_frequency"] = max(1, min(int(config["save_frequency"]), max_epochs))
        config["save_best_after"] = min(int(config["save_best_after"]), max_epochs)

    return agent_cfg, train_dir, experiment_name


def _maybe_freeze_running_stats(args: argparse.Namespace):
    if not args.freeze_input_running_stats and not args.freeze_value_running_stats:
        return None

    from rl_games.algos_torch import running_mean_std

    original_forward = running_mean_std.RunningMeanStd.forward

    def forward_with_optional_freeze(self, input, denorm=False, mask=None):
        insize = self.insize if isinstance(self.insize, tuple) else tuple(self.insize)
        is_value_stats = insize == (1,)
        should_freeze = (
            (args.freeze_value_running_stats and is_value_stats)
            or (args.freeze_input_running_stats and not is_value_stats)
        )
        if not should_freeze or not self.training:
            return original_forward(self, input, denorm=denorm, mask=mask)
        self.eval()
        try:
            return original_forward(self, input, denorm=denorm, mask=mask)
        finally:
            self.train()

    running_mean_std.RunningMeanStd.forward = forward_with_optional_freeze
    frozen = []
    if args.freeze_input_running_stats:
        frozen.append("input")
    if args.freeze_value_running_stats:
        frozen.append("value")
    _trace(f"freezing running stats during continuation: {', '.join(frozen)}")
    return original_forward


def _maybe_patch_behavior_anchor(args: argparse.Namespace):
    if args.behavior_anchor_coef <= 0.0:
        return None

    anchor_checkpoint = args.behavior_anchor_checkpoint or args.checkpoint
    if not anchor_checkpoint:
        raise RuntimeError("--behavior-anchor-coef > 0 requires --checkpoint or --behavior-anchor-checkpoint")

    from rl_games.algos_torch import a2c_continuous
    from rl_games.common import common_losses

    original_init = a2c_continuous.A2CAgent.__init__
    original_calc_gradients = a2c_continuous.A2CAgent.calc_gradients
    anchor_path = str(Path(anchor_checkpoint).expanduser())
    anchor_coef = float(args.behavior_anchor_coef)
    anchor_mode = args.behavior_anchor_mode
    hand_start_index = args.behavior_anchor_hand_start_index
    arm_weight = float(args.behavior_anchor_arm_weight)
    hand_weight = float(args.behavior_anchor_hand_weight)

    def init_with_behavior_anchor(self, base_name, params):
        original_init(self, base_name, params)
        checkpoint = torch_ext.safe_filesystem_op(torch.load, anchor_path, map_location=self.ppo_device)
        anchor_model = copy.deepcopy(self.model)
        anchor_model.load_state_dict(checkpoint["model"])
        anchor_model.to(self.ppo_device)
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)
        self.behavior_anchor_model = anchor_model
        self.behavior_anchor_coef = anchor_coef
        self.behavior_anchor_mode = anchor_mode
        self.behavior_anchor_hand_start_index = hand_start_index
        self.behavior_anchor_arm_weight = arm_weight
        self.behavior_anchor_hand_weight = hand_weight
        self.behavior_anchor_last_loss = torch.zeros((), device=self.ppo_device)
        self.behavior_anchor_last_metric = torch.zeros((), device=self.ppo_device)
        _trace(
            "loaded frozen behavior anchor: "
            f"path={anchor_path}, coef={anchor_coef:g}, mode={anchor_mode}, "
            f"hand_start={hand_start_index}, arm_weight={arm_weight:g}, hand_weight={hand_weight:g}"
        )

    def _weighted_mu_mse(self, mu: torch.Tensor, ref_mu: torch.Tensor) -> torch.Tensor:
        squared_error = (mu - ref_mu.detach()) ** 2
        if self.behavior_anchor_hand_start_index is not None and squared_error.shape[-1] > 0:
            weights = torch.full_like(squared_error, self.behavior_anchor_arm_weight)
            start = max(0, min(int(self.behavior_anchor_hand_start_index), squared_error.shape[-1]))
            if start < squared_error.shape[-1]:
                weights[..., start:] = self.behavior_anchor_hand_weight
            weighted = squared_error * weights
            normalizer = weights.mean(dim=-1).clamp_min(1e-6)
            return weighted.mean(dim=-1) / normalizer
        return squared_error.mean(dim=-1)

    def calc_gradients_with_behavior_anchor(self, input_dict):
        if not hasattr(self, "behavior_anchor_model"):
            return original_calc_gradients(self, input_dict)

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            anchor_batch_dict = {
                "is_train": True,
                "prev_actions": actions_batch,
                "obs": obs_batch,
            }
            if self.is_rnn:
                anchor_batch_dict["rnn_states"] = input_dict["rnn_states"]
                anchor_batch_dict["seq_length"] = self.seq_length
                if self.zero_rnn_on_done:
                    anchor_batch_dict["dones"] = input_dict["dones"]

            with torch.no_grad():
                ref_res_dict = self.behavior_anchor_model(anchor_batch_dict)
                ref_mu = ref_res_dict["mus"]
                ref_sigma = ref_res_dict["sigmas"]

            if self.behavior_anchor_mode == "kl":
                anchor_loss_per_sample = torch_ext.policy_kl(mu, sigma, ref_mu.detach(), ref_sigma.detach(), reduce=False)
            else:
                anchor_loss_per_sample = _weighted_mu_mse(self, mu, ref_mu)

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip
            )

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [
                    a_loss.unsqueeze(1),
                    c_loss,
                    entropy.unsqueeze(1),
                    b_loss.unsqueeze(1),
                    anchor_loss_per_sample.unsqueeze(1),
                ],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss, anchor_loss = losses

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
                + anchor_loss * self.behavior_anchor_coef
            )

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

            if self.behavior_anchor_mode == "kl":
                anchor_metric = anchor_loss.detach()
            else:
                anchor_metric = torch_ext.policy_kl(mu.detach(), sigma.detach(), ref_mu.detach(), ref_sigma.detach(), True)
            self.behavior_anchor_last_loss = anchor_loss.detach()
            self.behavior_anchor_last_metric = anchor_metric.detach()
            if getattr(self, "global_rank", 0) == 0:
                self.writer.add_scalar("losses/behavior_anchor", self.behavior_anchor_last_loss.item(), self.frame)
                self.writer.add_scalar("diagnostics/behavior_anchor_kl", self.behavior_anchor_last_metric.item(), self.frame)

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    a2c_continuous.A2CAgent.__init__ = init_with_behavior_anchor
    a2c_continuous.A2CAgent.calc_gradients = calc_gradients_with_behavior_anchor
    _trace(f"enabled behavior-anchor PPO patch: checkpoint={anchor_path}, coef={anchor_coef:g}, mode={anchor_mode}")
    return original_init, original_calc_gradients


def main() -> None:
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    _trace(f"loading task: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.dynamic_curriculum_alpha is not None and hasattr(
        env_cfg, "dynamic_grasp_speed_curriculum_override_alpha"
    ):
        env_cfg.dynamic_grasp_speed_curriculum_override_alpha = float(args_cli.dynamic_curriculum_alpha)
        _trace(f"overriding dynamic curriculum alpha: {env_cfg.dynamic_grasp_speed_curriculum_override_alpha:g}")
    if args_cli.tabletop_asset_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_asset_curriculum_override_alpha"
    ):
        env_cfg.tabletop_asset_curriculum_override_alpha = float(args_cli.tabletop_asset_curriculum_alpha)
        _trace(f"overriding tabletop asset curriculum alpha: {env_cfg.tabletop_asset_curriculum_override_alpha:g}")
    if args_cli.tabletop_motion_curriculum_alpha is not None and hasattr(
        env_cfg, "tabletop_motion_mode_curriculum_override_alpha"
    ):
        env_cfg.tabletop_motion_mode_curriculum_override_alpha = float(args_cli.tabletop_motion_curriculum_alpha)
        _trace(f"overriding tabletop motion curriculum alpha: {env_cfg.tabletop_motion_mode_curriculum_override_alpha:g}")
    if args_cli.tabletop_pregrasp_lead_time is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_lead_time"
    ):
        env_cfg.dynamic_tabletop_pregrasp_lead_time = float(args_cli.tabletop_pregrasp_lead_time)
        _trace(f"overriding tabletop pregrasp lead time: {env_cfg.dynamic_tabletop_pregrasp_lead_time:g}")
    if args_cli.tabletop_pregrasp_ahead_distance is not None and hasattr(
        env_cfg, "dynamic_tabletop_pregrasp_ahead_distance"
    ):
        env_cfg.dynamic_tabletop_pregrasp_ahead_distance = float(args_cli.tabletop_pregrasp_ahead_distance)
        _trace(
            "overriding tabletop pregrasp ahead distance: "
            f"{env_cfg.dynamic_tabletop_pregrasp_ahead_distance:g}"
        )
    if args_cli.scripted_action_prior_active_residual_scale is not None and hasattr(
        env_cfg, "scripted_action_prior_active_residual_scale"
    ):
        env_cfg.scripted_action_prior_active_residual_scale = float(
            args_cli.scripted_action_prior_active_residual_scale
        )
        _trace(
            "overriding scripted prior active residual scale: "
            f"{env_cfg.scripted_action_prior_active_residual_scale:g}"
        )
    if args_cli.scripted_relative_lift_target_scale is not None and hasattr(
        env_cfg, "scripted_tabletop_relative_lift_target_arm_delta"
    ):
        relative_scale = float(args_cli.scripted_relative_lift_target_scale)
        env_cfg.scripted_tabletop_relative_lift_target_arm_delta = tuple(
            relative_scale * float(value)
            for value in env_cfg.scripted_tabletop_relative_lift_target_arm_delta
        )
        _trace(f"scaling scripted relative lift target delta by: {relative_scale:g}")
    if args_cli.scripted_action_prior_lift_steps is not None and hasattr(
        env_cfg, "scripted_action_prior_lift_steps"
    ):
        env_cfg.scripted_action_prior_lift_steps = int(args_cli.scripted_action_prior_lift_steps)
        _trace(f"overriding scripted prior lift steps: {env_cfg.scripted_action_prior_lift_steps}")
    if args_cli.episode_length_s is not None and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
        _trace(f"overriding episode length: {env_cfg.episode_length_s:g} s")
    if args_cli.dynamic_success_hold_steps is not None and hasattr(
        env_cfg, "dynamic_success_hold_steps"
    ):
        env_cfg.dynamic_success_hold_steps = int(args_cli.dynamic_success_hold_steps)
        _trace(f"overriding dynamic success hold steps: {env_cfg.dynamic_success_hold_steps}")
    if args_cli.tabletop_post_success_hand_close_fraction is not None:
        env_cfg.tabletop_post_success_hand_close_fraction = float(
            args_cli.tabletop_post_success_hand_close_fraction
        )
        _trace(
            "overriding post-success hand close fraction: "
            f"{env_cfg.tabletop_post_success_hand_close_fraction:g}"
        )
    if args_cli.stability_target_latch_min_success_streak is not None:
        env_cfg.stability_target_latch_min_success_streak = int(
            args_cli.stability_target_latch_min_success_streak
        )
        _trace(
            "overriding stability target latch minimum streak: "
            f"{env_cfg.stability_target_latch_min_success_streak}"
        )

    _trace(f"making env: num_envs={env_cfg.scene.num_envs}, sim_device={env_cfg.sim.device}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    wandb_run = None
    try:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
        agent_cfg, train_dir, experiment_name = _configure_agent(agent_cfg, env.unwrapped, args_cli)
        config = agent_cfg["params"]["config"]
        env_block = agent_cfg["params"]["env"]
        rl_device = args_cli.rl_device or args_cli.device
        clip_obs = float(env_block.get("clip_observations", 100.0))
        clip_actions = float(env_block.get("clip_actions", 1.0))

        _trace(
            "wrapping env: "
            f"rl_device={rl_device}, clip_obs={clip_obs}, clip_actions={clip_actions}, "
            f"obs={env.unwrapped.cfg.observation_space}, actions={env.unwrapped.cfg.action_space}"
        )
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
        env.unwrapped.sim._app_control_on_stop_handle = None

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )

        run_dir = train_dir / experiment_name
        wandb_run = _maybe_init_wandb(args_cli, agent_cfg, run_dir, experiment_name)
        if wandb_run is not None:
            _trace(f"wandb run: {wandb_run.url or wandb_run.name}")

        _trace(
            "starting rl_games: "
            f"train_dir={train_dir}, run={experiment_name}, max_epochs={config['max_epochs']}, "
            f"horizon={config['horizon_length']}, minibatch={config['minibatch_size']}, "
            f"mini_epochs={config['mini_epochs']}"
        )
        original_rms_forward = _maybe_freeze_running_stats(args_cli)
        original_behavior_anchor_patch = _maybe_patch_behavior_anchor(args_cli)
        runner = Runner(algo_observer=IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()
        original_load_checkpoint = torch_ext.load_checkpoint

        def load_checkpoint_on_train_device(filename):
            print("=> loading checkpoint '{}'".format(filename))
            map_location = args_cli.rl_device or args_cli.device
            checkpoint = torch_ext.safe_filesystem_op(torch.load, filename, map_location=map_location)
            if args_cli.reset_optimizer_on_load:
                optimizer_state = checkpoint.get("optimizer")
                if not isinstance(optimizer_state, dict):
                    raise RuntimeError("--reset-optimizer-on-load requested, but checkpoint has no optimizer state")
                optimizer_state["state"] = {}
                lr = float(config["learning_rate"])
                for group in optimizer_state.get("param_groups", []):
                    group["lr"] = lr
                print(f"=> reset checkpoint optimizer state and set lr={lr:g}", flush=True)
            if args_cli.reset_epoch_on_load:
                checkpoint["epoch"] = 0
                checkpoint["frame"] = 0
                checkpoint["last_mean_rewards"] = -1000000000
                print("=> reset checkpoint epoch/frame/best-reward counters", flush=True)
            return checkpoint

        try:
            torch_ext.load_checkpoint = load_checkpoint_on_train_device
            runner.run({"train": True, "play": False, "checkpoint": args_cli.checkpoint, "sigma": args_cli.sigma})
        finally:
            torch_ext.load_checkpoint = original_load_checkpoint
            if original_rms_forward is not None:
                from rl_games.algos_torch import running_mean_std

                running_mean_std.RunningMeanStd.forward = original_rms_forward
            if original_behavior_anchor_patch is not None:
                from rl_games.algos_torch import a2c_continuous

                original_init, original_calc_gradients = original_behavior_anchor_patch
                a2c_continuous.A2CAgent.__init__ = original_init
                a2c_continuous.A2CAgent.calc_gradients = original_calc_gradients

        checkpoints = sorted((run_dir / "nn").glob("*.pth"))
        if not checkpoints:
            raise RuntimeError(f"Training finished but no checkpoint was written under {run_dir / 'nn'}")
        _trace(f"latest checkpoint: {checkpoints[-1]}")
        _trace(f"run directory: {run_dir}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
