#!/usr/bin/env bash
set -euo pipefail

ROOT=${SIMTOOLREAL_LAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${SIMTOOLREAL_LAB_PYTHON:-/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python}

TASK=${TASK:-SimToolReal-Revo2-Franka-StaticStableHoverJointTargetAB-Teacher-Direct-v0}
GPU_ID=${GPU_ID:-7}
NUM_ENVS=${NUM_ENVS:-512}
MAX_EPOCHS=${MAX_EPOCHS:-400}
SEED=${SEED:-42}
HORIZON_LENGTH=${HORIZON_LENGTH:-32}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
MINI_EPOCHS=${MINI_EPOCHS:-5}
SAVE_FREQUENCY=${SAVE_FREQUENCY:-25}
RUN_SUFFIX=${RUN_SUFFIX:-20260716}
WANDB_PROJECT=${WANDB_PROJECT:-simtoolreal_lab}
WANDB_ENTITY=${WANDB_ENTITY:-simonlsx}
WANDB_GROUP=${WANDB_GROUP:-static_state_encoder_ab_v12}
LOG_ROOT=${LOG_ROOT:-${ROOT}/logs/control_ablation/static_state_encoder_v12}
ENCODERS=${ENCODERS:-"flat structured"}

mkdir -p "${LOG_ROOT}/launcher_logs"
cd "${ROOT}"

for ENCODER in ${ENCODERS}; do
  EXPERIMENT_NAME="revo_joint_v12_${ENCODER}_seed${SEED}_${RUN_SUFFIX}"
  LOG_PATH="${LOG_ROOT}/launcher_logs/${EXPERIMENT_NAME}.log"
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/train_rl_games.py \
    --task "${TASK}" \
    --teacher-state-encoder "${ENCODER}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --max-epochs "${MAX_EPOCHS}" \
    --horizon-length "${HORIZON_LENGTH}" \
    --minibatch-size "${MINIBATCH_SIZE}" \
    --mini-epochs "${MINI_EPOCHS}" \
    --save-frequency "${SAVE_FREQUENCY}" \
    --log-root "${LOG_ROOT}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-name "${EXPERIMENT_NAME}" \
    --wandb-tags static teacher from_scratch v12 "${ENCODER}" semantic_encoder_ab \
    --wandb-mode online \
    --headless \
    --device cuda:0 \
    2>&1 | tee "${LOG_PATH}"
done
