#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/server_mvtec.yaml}"
ENV_FILE="${2:-${SERVER_ENV_FILE:-}}"
CLI_RUN_STAGE="${3:-}"
RUN_STAGE="${CLI_RUN_STAGE:-${RUN_STAGE:-all}}"
OVERRIDE_CANDIDATES=(
  DATASET
  DATA_ROOT
  FEW_SHOT_ROOT
  CHECKPOINT_PATH
  OUTPUT_ROOT
  DINOMALY_ROOT
  DEVICE
  RUN_MODE
  RUN_STAGE
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
  FEW_SHOT_ENHANCER_HIDDEN_DIM
  FEW_SHOT_ENHANCER_LR
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
  EVAL_FUSION_BETA_SWEEP
  EVAL_BETA_SELECTION_METRIC
)
OVERRIDE_NAMES=()
OVERRIDE_VALUES=()
ENV_FILE_NAMES=()

_name_in_candidates() {
  local needle="$1"
  local candidate
  for candidate in "${OVERRIDE_CANDIDATES[@]}"; do
    if [[ "${candidate}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

_join_names() {
  local IFS=,
  echo "$*"
}

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
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*= ]]; then
      name="${BASH_REMATCH[1]}"
      if _name_in_candidates "${name}"; then
        ENV_FILE_NAMES+=("${name}")
      fi
    fi
  done < "${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

for idx in "${!OVERRIDE_NAMES[@]}"; do
  export "${OVERRIDE_NAMES[$idx]}=${OVERRIDE_VALUES[$idx]}"
done
if [[ -n "${CLI_RUN_STAGE}" ]]; then
  RUN_STAGE="${CLI_RUN_STAGE}"
fi

RUN_MODE="${RUN_MODE:-smoke}"
DATASET="${DATASET:-mvtec}"
export RUN_MODE
export DATASET
export RUN_STAGE
LLM_DAS_RUNNER="$(basename "$0")"
LLM_DAS_CONFIG_PATH="${CONFIG_PATH}"
LLM_DAS_ENV_FILE="${ENV_FILE}"
LLM_DAS_STAGE_ARG="${CLI_RUN_STAGE}"
LLM_DAS_INLINE_OVERRIDES="$(_join_names "${OVERRIDE_NAMES[@]}")"
LLM_DAS_ENV_FILE_OVERRIDES="$(_join_names "${ENV_FILE_NAMES[@]}")"
export LLM_DAS_RUNNER
export LLM_DAS_CONFIG_PATH
export LLM_DAS_ENV_FILE
export LLM_DAS_STAGE_ARG
export LLM_DAS_INLINE_OVERRIDES
export LLM_DAS_ENV_FILE_OVERRIDES

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
if [[ -n "${LLM_DAS_ENV_FILE_OVERRIDES}" ]]; then
  echo "[llm-das-dinomaly] env_file_keys=${LLM_DAS_ENV_FILE_OVERRIDES}"
fi
if [[ -n "${LLM_DAS_INLINE_OVERRIDES}" ]]; then
  echo "[llm-das-dinomaly] inline_overrides=${LLM_DAS_INLINE_OVERRIDES}"
fi
echo "[llm-das-dinomaly] dataset=${DATASET}"
echo "[llm-das-dinomaly] mode=${RUN_MODE}"
echo "[llm-das-dinomaly] stage=${RUN_STAGE}"
echo "[llm-das-dinomaly] data=${DATA_ROOT:-<none>}"
if [[ -n "${FEW_SHOT_ROOT:-}" ]]; then
  echo "[llm-das-dinomaly] few_shot_root=${FEW_SHOT_ROOT}"
  echo "[llm-das-dinomaly] checkpoint=<few-shot-train-new>"
else
  echo "[llm-das-dinomaly] checkpoint=${CHECKPOINT_PATH}"
fi
echo "[llm-das-dinomaly] output=${OUTPUT_ROOT}"
echo "[llm-das-dinomaly] effective_config=${OUTPUT_ROOT}/effective_config.json"
echo "[llm-das-dinomaly] dinomaly=${DINOMALY_ROOT}"

python -m llm_das_dinomaly.pipelines.server_mvtec --config "${CONFIG_PATH}" --stage check
if [[ "${RUN_STAGE}" != "check" ]]; then
  python -m llm_das_dinomaly.pipelines.server_mvtec --config "${CONFIG_PATH}" --stage "${RUN_STAGE}"
fi
