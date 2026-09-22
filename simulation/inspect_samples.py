#!/usr/bin/env python3
"""Inspect ROOT calorimeter samples and emit a reproducible dataset summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot


REQUIRED_BRANCHES = {
    "cell_e",
    "cell_x",
    "cell_y",
    "cell_dx",
    "cell_dy",
    "cell_l",
    "ECAL1_e",
    "ECAL2_e",
    "ECAL3_e",
    "HCAL1_e",
    "HCAL2_e",
    "HCAL3_e",
    "Cal_e",
    "Caltotal_e",
    "particle_e",
    "particle_x",
    "particle_y",
    "particle_z",
    "particle_px",
    "particle_py",
    "particle_pz",
    "particle_pdgId",
}


def flattened(array: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(array, axis=None)).astype(np.float64)


def numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def inspect_file(path: Path) -> dict[str, Any]:
    with uproot.open(path) as root_file:
        if "physics" not in root_file:
            raise RuntimeError(f"{path} does not contain a 'physics' tree")
        tree = root_file["physics"]
        available = set(tree.keys())
        missing = sorted(REQUIRED_BRANCHES - available)
        num_entries = int(tree.num_entries)
        arrays = tree.arrays(sorted(REQUIRED_BRANCHES & available), library="ak")

    result: dict[str, Any] = {
        "file": path.name,
        "entries": num_entries,
        "missing_required_branches": missing,
    }

    for name in (
        "particle_e",
        "particle_pdgId",
        "particle_x",
        "particle_y",
        "particle_z",
        "particle_px",
        "particle_py",
        "particle_pz",
        "Cal_e",
        "Caltotal_e",
    ):
        if name in arrays.fields:
            result[name] = numeric_summary(flattened(arrays[name]))

    if "cell_e" in arrays.fields:
        counts = ak.to_numpy(ak.num(arrays["cell_e"], axis=1)).astype(np.float64)
        result["cells_per_event"] = numeric_summary(counts)

    if "cell_l" in arrays.fields:
        layers = arrays["cell_l"]
        result["cells_per_event_by_layer"] = {
            str(layer): numeric_summary(
                ak.to_numpy(ak.sum(layers == layer, axis=1)).astype(np.float64)
            )
            for layer in (1, 2, 3)
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.data_dir.glob("*.root"))
    if not paths:
        raise SystemExit(f"No ROOT files found in {args.data_dir}")

    report = {
        "data_directory": str(args.data_dir.resolve()),
        "files": [inspect_file(path) for path in paths],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
