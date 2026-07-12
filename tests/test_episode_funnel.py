import torch

from simtoolreal_lab.teacher_student.evaluation import (
    EpisodeFunnelAccumulator,
    EpisodeFunnelTracker,
    flatten_numeric_metrics,
)


def test_flatten_numeric_metrics_skips_non_numeric_leaves():
    metrics = flatten_numeric_metrics(
        {
            "success_rate": 0.5,
            "passed": True,
            "failure_funnel": {
                "episodes": 8,
                "available_signals": ["success"],
                "conversion_rates": {"lift_to_stable_hold": 0.75},
            },
        },
        "eval",
    )

    assert metrics == {
        "eval/success_rate": 0.5,
        "eval/passed": 1.0,
        "eval/failure_funnel/episodes": 8.0,
        "eval/failure_funnel/conversion_rates/lift_to_stable_hold": 0.75,
    }


def _extras(**values):
    return {f"{name}_env": torch.tensor(value) for name, value in values.items()}


def test_rolling_failure_funnel_assigns_one_primary_outcome():
    tracker = EpisodeFunnelTracker(4, "cpu")
    accumulator = EpisodeFunnelAccumulator("dynamic_tabletop_grasp")
    tracker.update(
        _extras(
            palm_reach=[False, True, True, True],
            strict_thumb_contact=[False, True, True, True],
            strict_opposing_contact=[False, False, True, True],
            strict_true_grasp=[False, False, True, True],
            lifted=[False, False, False, True],
            stable_hold=[False, False, False, True],
            success=[False, False, False, True],
            success_streak=[0, 0, 2, 8],
        )
    )
    accumulator.add(tracker.snapshot(torch.arange(4)))
    summary = accumulator.summary()

    assert summary["primary_failure_counts"] == {
        "insufficient_success_streak": 0,
        "no_lift": 1,
        "no_palm_reach": 1,
        "no_strict_opposition": 1,
        "no_strict_thumb_contact": 0,
        "no_strict_grasp": 0,
        "no_stable_hold": 0,
        "success": 1,
    }
    assert sum(summary["primary_failure_counts"].values()) == 4
    assert summary["conversion_rates"]["strict_grasp_to_lift"] == 0.5


def test_tracker_accumulates_stages_and_resets_selected_envs():
    tracker = EpisodeFunnelTracker(2, "cpu")
    tracker.update(
        _extras(
            strict_true_grasp=[True, False],
            success_streak=[2, 1],
            tabletop_arm_lift_baseline_latched=[True, False],
            tabletop_lift_action_prior_rew=[0.2, 0.1],
        )
    )
    tracker.update(
        _extras(
            strict_true_grasp=[False, True],
            success_streak=[1, 3],
            tabletop_arm_lift_baseline_latched=[False, True],
            tabletop_lift_action_prior_rew=[0.1, 0.6],
        )
    )
    snapshot = tracker.snapshot(torch.arange(2))

    assert snapshot["boolean"]["strict_true_grasp"].tolist() == [True, True]
    assert snapshot["boolean"]["tabletop_arm_lift_baseline_latched"].tolist() == [True, True]
    assert snapshot["maximum"]["success_streak"].tolist() == [2.0, 3.0]
    torch.testing.assert_close(
        snapshot["maximum"]["tabletop_lift_action_prior_rew"],
        torch.tensor([0.2, 0.6]),
    )

    tracker.reset(torch.tensor([0]))
    assert tracker.boolean["strict_true_grasp"].tolist() == [False, True]
    assert tracker.boolean["tabletop_arm_lift_baseline_latched"].tolist() == [False, True]
    assert tracker.maximum["success_streak"].tolist() == [0.0, 3.0]
    torch.testing.assert_close(
        tracker.maximum["tabletop_lift_action_prior_rew"],
        torch.tensor([0.0, 0.6]),
    )
