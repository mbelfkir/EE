#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PHOTON_PATH="${1:-${PROJECT_DIR}/data/photon_30GeV_10k.root}"
ELECTRON_PATH="${2:-${PROJECT_DIR}/data/electron_30GeV_10k.root}"
OUTPUT_DIR="${3:-${PROJECT_DIR}/outputs/plots}"

command -v root >/dev/null 2>&1 || { echo "CERN ROOT is not available on PATH" >&2; exit 1; }
mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
root -l -b -q "compute_layer2_shower_shapes.C(\"${PHOTON_PATH}\",\"${ELECTRON_PATH}\",\"${OUTPUT_DIR}\")"
