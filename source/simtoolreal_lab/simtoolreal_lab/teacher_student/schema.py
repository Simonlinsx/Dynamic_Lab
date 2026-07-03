"""Shared data contracts for the dynamic grasp teacher-student pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskFamily(str, Enum):
    """Dynamic dexterous grasp task families."""

    DYNAMIC_TABLETOP_GRASP = "dynamic_tabletop_grasp"
    FALLING_BATON_GRASP = "falling_baton_grasp"


class HandEmbodiment(str, Enum):
    """Supported dexterous hand embodiments."""

    REVO2 = "revo2"
    INSPIRE = "inspire"


@dataclass(frozen=True)
class ActionContract:
    """Meaning and dimensionality of a policy action vector."""

    name: str
    hand: HandEmbodiment
    action_dim: int
    arm_dim: int
    hand_dim: int
    normalized: bool
    description: str


ACTION_CONTRACTS: dict[str, ActionContract] = {
    "revo2_semantic_13d": ActionContract(
        name="revo2_semantic_13d",
        hand=HandEmbodiment.REVO2,
        action_dim=13,
        arm_dim=7,
        hand_dim=6,
        normalized=True,
        description="7 Franka arm deltas/targets + 6 Revo2 semantic hand fractions.",
    ),
    "revo2_physical_18d": ActionContract(
        name="revo2_physical_18d",
        hand=HandEmbodiment.REVO2,
        action_dim=18,
        arm_dim=7,
        hand_dim=11,
        normalized=True,
        description="Legacy V699 sim-only contract: 7 Franka arm targets + 11 Revo2 physical hand targets.",
    ),
    "inspire_physical_19d": ActionContract(
        name="inspire_physical_19d",
        hand=HandEmbodiment.INSPIRE,
        action_dim=19,
        arm_dim=7,
        hand_dim=12,
        normalized=True,
        description="Legacy sim-only contract: 7 Franka arm targets + 12 Inspire hand joint targets.",
    ),
    "inspire_semantic_13d": ActionContract(
        name="inspire_semantic_13d",
        hand=HandEmbodiment.INSPIRE,
        action_dim=13,
        arm_dim=7,
        hand_dim=6,
        normalized=True,
        description="7 Franka arm targets + 6 Inspire active hand commands.",
    ),
}


@dataclass(frozen=True)
class TensorSpec:
    """Expected tensor shape, excluding the leading batch dimension."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "float32"
    required: bool = True


@dataclass(frozen=True)
class StudentDatasetSpec:
    """Schema for student pretraining and teacher-student rollout datasets."""

    task_family: TaskFamily
    hand: HandEmbodiment
    action_contract: ActionContract
    history: int = 4
    num_object_points: int = 128
    proprio_dim: int = 76
    compact_privileged_dim: int = 32
    num_fingertips: int = 5
    target_key: str = "target"
    success_field: str = "episode_success"
    tensors: tuple[TensorSpec, ...] = field(default_factory=tuple)

    @property
    def action_dim(self) -> int:
        return self.action_contract.action_dim

    def with_tensor_specs(self) -> "StudentDatasetSpec":
        """Return a copy with the standard tensor specs populated."""

        tensors = (
            TensorSpec("pointcloud_seq", (self.history, self.num_object_points, 3)),
            TensorSpec("pointcloud_valid_seq", (self.history, self.num_object_points)),
            TensorSpec("proprio_seq", (self.history, self.proprio_dim)),
            TensorSpec(self.target_key, (self.action_dim,)),
            TensorSpec("point_flow_velocity", (self.num_object_points, 3), required=False),
            TensorSpec("point_flow_delta", (self.num_object_points, 3), required=False),
            TensorSpec("affordance_region_labels", (self.num_object_points,)),
            TensorSpec("compact_privileged", (self.compact_privileged_dim,)),
            TensorSpec("hold_target", (self.action_contract.hand_dim,), required=False),
            TensorSpec("hold_mask", (), dtype="float32", required=False),
            TensorSpec("phase", (), dtype="int64", required=False),
            TensorSpec(self.success_field, (), dtype="float32", required=False),
        )
        return StudentDatasetSpec(
            task_family=self.task_family,
            hand=self.hand,
            action_contract=self.action_contract,
            history=self.history,
            num_object_points=self.num_object_points,
            proprio_dim=self.proprio_dim,
            compact_privileged_dim=self.compact_privileged_dim,
            num_fingertips=self.num_fingertips,
            target_key=self.target_key,
            success_field=self.success_field,
            tensors=tensors,
        )


def default_dataset_spec(
    task_family: TaskFamily | str = TaskFamily.FALLING_BATON_GRASP,
    hand: HandEmbodiment | str = HandEmbodiment.REVO2,
    action_contract: str = "revo2_semantic_13d",
    history: int = 4,
    num_object_points: int = 128,
    proprio_dim: int = 76,
    compact_privileged_dim: int = 32,
) -> StudentDatasetSpec:
    """Build the default V699-style dataset spec."""

    task_family = TaskFamily(task_family)
    hand = HandEmbodiment(hand)
    contract = ACTION_CONTRACTS[action_contract]
    if contract.hand != hand:
        raise ValueError(f"Action contract {action_contract!r} is for {contract.hand}, not {hand}.")
    return StudentDatasetSpec(
        task_family=task_family,
        hand=hand,
        action_contract=contract,
        history=history,
        num_object_points=num_object_points,
        proprio_dim=proprio_dim,
        compact_privileged_dim=compact_privileged_dim,
    ).with_tensor_specs()


STUDENT_REQUIRED_KEYS = (
    "pointcloud_seq",
    "pointcloud_valid_seq",
    "proprio_seq",
    "target",
    "affordance_region_labels",
    "compact_privileged",
)


def _shape_tail(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    if len(shape) == 0:
        return ()
    return tuple(int(dim) for dim in shape[1:])


def validate_student_batch(batch: dict[str, Any], spec: StudentDatasetSpec | None = None) -> list[str]:
    """Validate a student batch/dataset dict and return human-readable errors."""

    spec = spec or default_dataset_spec()
    errors: list[str] = []

    for key in STUDENT_REQUIRED_KEYS:
        if key not in batch:
            errors.append(f"missing required key: {key}")

    flow_keys = ("point_flow_velocity", "point_flow_delta")
    if not any(key in batch for key in flow_keys):
        errors.append("missing one flow target: point_flow_velocity or point_flow_delta")

    for tensor_spec in spec.tensors:
        if tensor_spec.name not in batch:
            if tensor_spec.required:
                errors.append(f"missing required tensor: {tensor_spec.name}")
            continue
        tail = _shape_tail(batch[tensor_spec.name])
        if tail is None:
            errors.append(f"{tensor_spec.name} has no shape attribute")
            continue
        expected = tuple(dim for dim in tensor_spec.shape if isinstance(dim, int))
        if len(tensor_spec.shape) != len(tail):
            errors.append(f"{tensor_spec.name} rank mismatch: expected tail {tensor_spec.shape}, got {tail}")
            continue
        concrete_expected = tuple(
            int(dim) if isinstance(dim, int) else tail[index] for index, dim in enumerate(tensor_spec.shape)
        )
        if tail != concrete_expected:
            errors.append(f"{tensor_spec.name} shape mismatch: expected tail {concrete_expected}, got {tail}")

    return errors
