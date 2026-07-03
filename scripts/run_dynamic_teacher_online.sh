#!/usr/bin/env bash
set -euo pipefail

ROOT=${SIMTOOLREAL_LAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${SIMTOOLREAL_LAB_PYTHON:-/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python}

TASK=${TASK:-SimToolReal-Revo2-Franka-FallingBatonEasy-Teacher-Direct-v0}
GPU_ID=${GPU_ID:-6}
NUM_ENVS=${NUM_ENVS:-512}
MAX_EPOCHS=${MAX_EPOCHS:-1200}
SEED=${SEED:-42}
HORIZON_LENGTH=${HORIZON_LENGTH:-32}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
MINI_EPOCHS=${MINI_EPOCHS:-5}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-50}
CHECKPOINT=${CHECKPOINT:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-revo2_falling_baton_easy_teacher_metrics_online_512env_1200ep_20260701}
WANDB_PROJECT=${WANDB_PROJECT:-simtoolreal_lab}
WANDB_ENTITY=${WANDB_ENTITY:-simonlsx}
WANDB_GROUP=${WANDB_GROUP:-dynamic_dexterous_grasp_teacher}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${EXPERIMENT_NAME}}

LOG_DIR="${ROOT}/logs/run_scripts"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXPERIMENT_NAME}.log"

cd "${ROOT}"
echo "[RUN] task=${TASK}"
echo "[RUN] gpu=${GPU_ID} num_envs=${NUM_ENVS} max_epochs=${MAX_EPOCHS} seed=${SEED}"
echo "[RUN] checkpoint=${CHECKPOINT:-<none>}"
echo "[RUN] wandb=${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "[RUN] log=${LOG_PATH}"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

TASK_TAG="dynamic_grasp"
if [[ "${TASK}" == *"DynamicTabletop"* ]]; then
  TASK_TAG="tabletop"
elif [[ "${TASK}" == *"FallingBaton"* ]]; then
  TASK_TAG="falling_baton"
fi

HAND_TAG="revo2"
if [[ "${TASK}" == *"Inspire"* ]]; then
  HAND_TAG="inspire"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/train_rl_games.py \
  --task "${TASK}" \
  --num-envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --max-epochs "${MAX_EPOCHS}" \
  --horizon-length "${HORIZON_LENGTH}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --mini-epochs "${MINI_EPOCHS}" \
  --save-frequency "${SAVE_FREQUENCY}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  "${CHECKPOINT_ARGS[@]}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-tags dynamic_teacher "${HAND_TAG}" franka "${TASK_TAG}" isaaclab privileged_rl \
  --wandb-mode online \
  --headless \
  --device cuda:0 \
  2>&1 | tee "${LOG_PATH}"
