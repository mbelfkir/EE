#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v root >/dev/null 2>&1 || { echo "CERN ROOT is not available on PATH" >&2; exit 1; }
mkdir -p "${PROJECT_DIR}/outputs/plots"

cd "${SCRIPT_DIR}"
root -l -b -q 'plot_basic_shower_comparison.C("../data","../outputs/plots")'
