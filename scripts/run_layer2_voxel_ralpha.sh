#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${1:-${PROJECT_DIR}/data/photon_30GeV_10k.root}"
OUTPUT_PREFIX="${2:-${PROJECT_DIR}/outputs/plots/layer2_voxel_ralpha_photon_30GeV}"

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

python3 "${SCRIPT_DIR}/plot_layer2_voxel_ralpha.py" \
  --input "${INPUT_PATH}" \
  --output-prefix "${OUTPUT_PREFIX}"
