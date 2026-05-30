#!/usr/bin/env bash
set -euo pipefail

# Placeholder runner for the official Dinomaly baseline.
# Fill DINOMALY_ROOT and DATA_ROOT in your local visual environment, then call
# the official dataset-specific script. Keep this separate from the LLM env.

: "${DINOMALY_ROOT:?Set DINOMALY_ROOT to the official Dinomaly checkout}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
: "${DATASET:=mvtec}"

cd "${DINOMALY_ROOT}"

case "${DATASET}" in
  mvtec)
    python dinomaly_mvtec_uni.py --data_path "${DATA_ROOT}"
    ;;
  visa)
    python dinomaly_visa_uni.py --data_path "${DATA_ROOT}"
    ;;
  realiad)
    python dinomaly_realiad_uni.py --data_path "${DATA_ROOT}"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}; expected mvtec, visa, or realiad" >&2
    exit 2
    ;;
esac
