#!/usr/bin/env bash
set -euo pipefail

ROOT=${SIMTOOLREAL_LAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${SIMTOOLREAL_LAB_PYTHON:-/data1/linsixu/miniconda3/envs/dynamicvla-isaac/bin/python}

TASK=${TASK:-SimToolReal-Revo2-Franka-FallingBatonEasy-Teacher-Direct-v0}
GPU_ID=${GPU_ID:-6}
RUN_NAME=${RUN_NAME:-revo2_falling_baton_easy_teacher_metrics_online_512env_1200ep_20260701}
CHECKPOINT=${CHECKPOINT:-}
NUM_ENVS=${NUM_ENVS:-256}
EPISODES=${EPISODES:-256}
SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-0.30}
SAVE_SUCCESS_VIDEOS=${SAVE_SUCCESS_VIDEOS:-1}
VIDEO_ATTEMPTS=${VIDEO_ATTEMPTS:-30}
VIDEO_ENVS=${VIDEO_ENVS:-4}
VIDEO_POST_SUCCESS_STEPS=${VIDEO_POST_SUCCESS_STEPS:-24}
VIDEO_CAMERA_TRACK_OFFSET=${VIDEO_CAMERA_TRACK_OFFSET:-"0.65 0.95 0.65"}
VIDEO_CAMERA_TRACK_TARGET_OFFSET=${VIDEO_CAMERA_TRACK_TARGET_OFFSET:-"0.0 0.0 0.05"}
VIDEO_CAMERA_FOCAL_LENGTH=${VIDEO_CAMERA_FOCAL_LENGTH:-24.0}
VIDEO_CAMERA_RESOLUTION=${VIDEO_CAMERA_RESOLUTION:-"960 540"}
WANDB_PROJECT=${WANDB_PROJECT:-simtoolreal_lab}
WANDB_ENTITY=${WANDB_ENTITY:-simonlsx}
WANDB_GROUP=${WANDB_GROUP:-dynamic_dexterous_grasp_teacher_eval}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-eval_${RUN_NAME}}

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT=$(find "${ROOT}/logs/rl_games/${RUN_NAME}/nn" -maxdepth 1 -type f -name "*.pth" | sort | tail -n 1)
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[EVAL-RUN] no checkpoint found for RUN_NAME=${RUN_NAME}" >&2
  exit 2
fi

OUTPUT_DIR="${ROOT}/outputs/eval_rl_games/${RUN_NAME}_eval_online"
LOG_DIR="${ROOT}/logs/run_scripts"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_eval.log"

cd "${ROOT}"
echo "[EVAL-RUN] task=${TASK}"
echo "[EVAL-RUN] checkpoint=${CHECKPOINT}"
echo "[EVAL-RUN] output=${OUTPUT_DIR}"
echo "[EVAL-RUN] log=${LOG_PATH}"
echo "[EVAL-RUN] video_camera_track_offset=${VIDEO_CAMERA_TRACK_OFFSET}"
echo "[EVAL-RUN] video_camera_track_target_offset=${VIDEO_CAMERA_TRACK_TARGET_OFFSET}"

read -r VIDEO_CAMERA_TRACK_OFFSET_X VIDEO_CAMERA_TRACK_OFFSET_Y VIDEO_CAMERA_TRACK_OFFSET_Z <<< "${VIDEO_CAMERA_TRACK_OFFSET}"
read -r VIDEO_CAMERA_TRACK_TARGET_OFFSET_X VIDEO_CAMERA_TRACK_TARGET_OFFSET_Y VIDEO_CAMERA_TRACK_TARGET_OFFSET_Z <<< "${VIDEO_CAMERA_TRACK_TARGET_OFFSET}"
read -r VIDEO_CAMERA_WIDTH VIDEO_CAMERA_HEIGHT <<< "${VIDEO_CAMERA_RESOLUTION}"

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

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" scripts/evaluate_rl_games.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --num-envs "${NUM_ENVS}" \
  --episodes "${EPISODES}" \
  --success-threshold "${SUCCESS_THRESHOLD}" \
  --save-success-videos "${SAVE_SUCCESS_VIDEOS}" \
  --video-attempts "${VIDEO_ATTEMPTS}" \
  --video-envs "${VIDEO_ENVS}" \
  --video-stride 2 \
  --video-fps 30 \
  --video-post-success-steps "${VIDEO_POST_SUCCESS_STEPS}" \
  --video-camera-track-offset "${VIDEO_CAMERA_TRACK_OFFSET_X}" "${VIDEO_CAMERA_TRACK_OFFSET_Y}" "${VIDEO_CAMERA_TRACK_OFFSET_Z}" \
  --video-camera-track-target-offset "${VIDEO_CAMERA_TRACK_TARGET_OFFSET_X}" "${VIDEO_CAMERA_TRACK_TARGET_OFFSET_Y}" "${VIDEO_CAMERA_TRACK_TARGET_OFFSET_Z}" \
  --video-camera-focal-length "${VIDEO_CAMERA_FOCAL_LENGTH}" \
  --video-camera-resolution "${VIDEO_CAMERA_WIDTH}" "${VIDEO_CAMERA_HEIGHT}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-tags dynamic_teacher_eval "${HAND_TAG}" franka "${TASK_TAG}" isaaclab success_video \
  --wandb-mode online \
  --output-dir "${OUTPUT_DIR}" \
  --headless \
  --device cuda:0 \
  2>&1 | tee "${LOG_PATH}"
