#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/server_mvtec.yaml}"
ENV_FILE="${2:-${SERVER_ENV_FILE:-}}"
OVERRIDE_CANDIDATES=(
  DATASET
  DATA_ROOT
  FEW_SHOT_ROOT
  CHECKPOINT_PATH
  OUTPUT_ROOT
  DINOMALY_ROOT
  DEVICE
  RUN_MODE
  BATCH_SIZE
  MVTEC_CATEGORY
  MPDD_CATEGORY
  VISA_CATEGORY
  BASE_TRAIN_IF_MISSING
  BASE_FORCE_RETRAIN
  BASE_TOTAL_ITERS
  BASE_EVAL_INTERVAL
  BASE_CHECKPOINT_DIR
  FEW_SHOT_BASE_TOTAL_ITERS
  FEW_SHOT_BASE_EVAL_INTERVAL
  FEW_SHOT_ENHANCER_EPOCHS
  MAX_SAMPLES
  SEARCH_BUDGET
  HARD_SAMPLE_SHARD_SIZE
  CACHE_IMAGES
  REGENERATE_HARD_SAMPLES
  RETRAIN_ENHANCER
  ENHANCER_EPOCHS
  ENHANCER_HIDDEN_DIM
  ENHANCER_LR
  PROGRESS
  EVAL_ENABLED
  EVAL_BATCH_SIZE
  EVAL_NUM_WORKERS
  EVAL_RESIZE_MASK
  EVAL_LIMIT_PER_CATEGORY
  EVAL_PIXEL_METRICS
  EVAL_PIXEL_AUPRO
  EVAL_EPOCH_PIXEL_METRICS
  EVAL_FUSION_BETA
)
OVERRIDE_NAMES=()
OVERRIDE_VALUES=()

for name in "${OVERRIDE_CANDIDATES[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    OVERRIDE_NAMES+=("${name}")
    OVERRIDE_VALUES+=("${!name}")
  fi
done

if [[ -z "${ENV_FILE}" && -f "configs/server_paths.env" ]]; then
  ENV_FILE="configs/server_paths.env"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file not found: ${ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

for idx in "${!OVERRIDE_NAMES[@]}"; do
  export "${OVERRIDE_NAMES[$idx]}=${OVERRIDE_VALUES[$idx]}"
done

RUN_MODE="${RUN_MODE:-smoke}"
DATASET="${DATASET:-mvtec}"
export RUN_MODE
export DATASET

if [[ -z "${DATA_ROOT:-}" && -z "${FEW_SHOT_ROOT:-}" ]]; then
  echo "Set DATA_ROOT to the MVTec root or FEW_SHOT_ROOT to a complete few-shot root" >&2
  exit 2
fi
if [[ -z "${FEW_SHOT_ROOT:-}" ]]; then
  : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a trained Dinomaly checkpoint .pth, or set FEW_SHOT_ROOT to train a new base checkpoint}"
fi
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable output directory}"
: "${DINOMALY_ROOT:=third_party/Dinomaly}"
export DINOMALY_ROOT

if [[ ! -d "${DINOMALY_ROOT}" ]]; then
  echo "Dinomaly root not found: ${DINOMALY_ROOT}" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 2
fi

if [[ "${RUN_MODE}" == "smoke" ]]; then
  export MVTEC_CATEGORY="${MVTEC_CATEGORY:-bottle}"
fi

echo "[llm-das-dinomaly] config=${CONFIG_PATH}"
if [[ -n "${ENV_FILE}" ]]; then
  echo "[llm-das-dinomaly] env_file=${ENV_FILE}"
fi
echo "[llm-das-dinomaly] dataset=${DATASET}"
echo "[llm-das-dinomaly] mode=${RUN_MODE}"
echo "[llm-das-dinomaly] data=${DATA_ROOT:-<none>}"
if [[ -n "${FEW_SHOT_ROOT:-}" ]]; then
  echo "[llm-das-dinomaly] few_shot_root=${FEW_SHOT_ROOT}"
  echo "[llm-das-dinomaly] checkpoint=<few-shot-train-new>"
else
  echo "[llm-das-dinomaly] checkpoint=${CHECKPOINT_PATH}"
fi
echo "[llm-das-dinomaly] output=${OUTPUT_ROOT}"
echo "[llm-das-dinomaly] dinomaly=${DINOMALY_ROOT}"

python -m llm_das_dinomaly.pipelines.server_mvtec --config "${CONFIG_PATH}" --stage check
python -m llm_das_dinomaly.pipelines.server_mvtec --config "${CONFIG_PATH}" --stage all
