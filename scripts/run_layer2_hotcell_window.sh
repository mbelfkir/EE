#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_PATH="${1:-${PROJECT_DIR}/data/photon_30GeV_10k.root}"
OUTPUT_PATH="${2:-${PROJECT_DIR}/outputs/plots/layer2_hotcell_window_photon_30GeV.png}"

command -v root >/dev/null 2>&1 || { echo "CERN ROOT is not available on PATH" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${SCRIPT_DIR}"
root -l -b -q "plot_layer2_hotcell_window.C(\"${INPUT_PATH}\",\"${OUTPUT_PATH}\")"
