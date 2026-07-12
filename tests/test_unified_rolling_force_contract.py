import ast
from pathlib import Path


CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "simtoolreal_lab"
    / "simtoolreal_lab"
    / "tasks"
    / "dynamic_dexterous_grasp"
    / "dynamic_dexterous_grasp_env_cfg.py"
)


def _class_constants(class_name: str) -> dict[str, object]:
    tree = ast.parse(CFG_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    constants: dict[str, object] = {}
    for statement in node.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            constants[target.id] = ast.literal_eval(statement.value)
    return constants


def test_stage3_force_contacts_are_diagnostics_not_reward_or_success_gates():
    values = _class_constants("_UnifiedRollingLiftHoldStage3Contract")

    assert values["object_contact_force_diagnostics_enabled"] is True
    assert values["tabletop_force_grasp_rew_scale"] == 0.0
    assert values["tabletop_force_grasp_streak_rew_scale"] == 0.0
    assert values["tabletop_force_stable_grasp_rew_scale"] == 0.0
    assert values["tabletop_force_grasp_loss_penalty_scale"] == 0.0
    assert values["tabletop_lift_rewards_require_force_grasp"] is False
    assert values["tabletop_lift_gate_requires_force_grasp"] is False
    assert values["tabletop_no_lift_uses_force_grasp_gate"] is False
    assert values["tabletop_success_requires_force_grasp"] is False
    assert values["tabletop_arm_lift_progress_baseline_mode"] == "first_strict_grasp"
