from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    import awkward as ak  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import uproot
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "plot_layer2_voxel_ralpha.py requires uproot, awkward, numpy, and matplotlib."
    ) from exc


MM_PER_UNIT = 1440.0
PROJECT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full layer-2 voxel representation in (r, alpha).")
    parser.add_argument(
        "--input",
        default=str(PROJECT_DIR / "data" / "photon_30GeV_10k.root"),
    )
    parser.add_argument(
        "--output-prefix",
        default=str(PROJECT_DIR / "outputs" / "plots" / "layer2_voxel_ralpha_full_photon_30GeV"),
    )
    parser.add_argument("--max-events", type=int, default=0, help="0 means all events.")
    return parser.parse_args()


def x_to_eta(x_mm: float) -> float:
    return x_mm / MM_PER_UNIT


def y_to_phi(y_mm: float) -> float:
    return y_mm / MM_PER_UNIT


def build_event_voxels(cell_e, cell_x, cell_y, cell_l) -> list[dict[str, float]]:
    hot_energy = None
    hot_eta = 0.0
    hot_phi = 0.0

    for energy, x_mm, y_mm, layer in zip(cell_e, cell_x, cell_y, cell_l):
        if int(layer) != 2:
            continue
        energy = float(energy)
        if hot_energy is None or energy > hot_energy:
            hot_energy = energy
            hot_eta = x_to_eta(float(x_mm))
            hot_phi = y_to_phi(float(y_mm))

    if hot_energy is None:
        return []

    voxels: list[dict[str, float]] = []
    for energy, x_mm, y_mm, layer in zip(cell_e, cell_x, cell_y, cell_l):
        if int(layer) != 2:
            continue
        deta = x_to_eta(float(x_mm)) - hot_eta
        dphi = y_to_phi(float(y_mm)) - hot_phi
        r = math.sqrt(deta * deta + dphi * dphi)
        alpha = math.atan2(dphi, deta) if (deta != 0.0 or dphi != 0.0) else 0.0
        voxels.append(
            {
                "delta_eta": deta,
                "delta_phi": dphi,
                "r": r,
                "alpha": alpha,
                "energy": float(energy),
            }
        )
    return voxels


def aggregate_mean_voxels(events_voxels: list[list[dict[str, float]]]) -> list[dict[str, float | int]]:
    accumulator: dict[tuple[float, float], dict[str, float]] = defaultdict(lambda: {"sum_energy": 0.0, "count": 0.0})

    for voxels in events_voxels:
        for voxel in voxels:
            key = (voxel["delta_eta"], voxel["delta_phi"])
            accumulator[key]["sum_energy"] += float(voxel["energy"])
            accumulator[key]["count"] += 1.0

    rows: list[dict[str, float | int]] = []
    for flat_index, ((deta, dphi), values) in enumerate(sorted(accumulator.items(), key=lambda item: (item[0][1], item[0][0]))):
        r = math.sqrt(deta * deta + dphi * dphi)
        alpha = math.atan2(dphi, deta) if (deta != 0.0 or dphi != 0.0) else 0.0
        rows.append(
            {
                "flat_index": flat_index,
                "delta_eta": deta,
                "delta_phi": dphi,
                "r": r,
                "alpha": alpha,
                "mean_energy": values["sum_energy"] / max(values["count"], 1.0),
                "event_occupancy": int(values["count"]),
            }
        )
    return rows


def save_voxel_table(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "flat_index",
                "delta_eta",
                "delta_phi",
                "r",
                "alpha",
                "mean_energy",
                "event_occupancy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_voxels(rows: list[dict[str, float | int]], output_png: Path, title: str) -> None:
    alpha = np.asarray([float(row["alpha"]) for row in rows], dtype=np.float32)
    radius = np.asarray([float(row["r"]) for row in rows], dtype=np.float32)
    energy = np.asarray([float(row["mean_energy"]) for row in rows], dtype=np.float32)
    flat_index = np.asarray([int(row["flat_index"]) for row in rows], dtype=np.int32)

    if np.allclose(energy.max(), energy.min()):
        marker_size = np.full_like(energy, 120.0)
    else:
        marker_size = 40.0 + 280.0 * (energy - energy.min()) / (energy.max() - energy.min())

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    polar_ax = fig.add_subplot(1, 2, 2, projection="polar")

    scatter = ax.scatter(alpha, radius, c=energy, s=marker_size, cmap="viridis", edgecolors="black", linewidths=0.35)
    for x, y, idx in zip(alpha, radius, flat_index):
        ax.text(x, y, f"{idx}", fontsize=5, ha="center", va="center")
    ax.set_xlabel("alpha [rad]")
    ax.set_ylabel("r [delta eta-phi units]")
    ax.set_title("Full layer-2 voxel centers in (alpha, r)")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, shrink=0.86, label="Mean deposited energy")

    polar = polar_ax.scatter(alpha, radius, c=energy, s=marker_size, cmap="viridis", edgecolors="black", linewidths=0.35)
    polar_ax.set_title("Polar voxel view")
    polar_ax.grid(alpha=0.25)
    fig.colorbar(polar, ax=polar_ax, shrink=0.86, label="Mean deposited energy")

    fig.suptitle(title)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)

    with uproot.open(input_path) as root_file:
        tree = root_file["physics"]
        arrays = tree.arrays(["cell_e", "cell_x", "cell_y", "cell_l"], library="ak")

    max_events = len(arrays["cell_e"]) if args.max_events <= 0 else min(args.max_events, len(arrays["cell_e"]))
    events_voxels: list[list[dict[str, float]]] = []

    for event_index in range(max_events):
        voxels = build_event_voxels(
            arrays["cell_e"][event_index],
            arrays["cell_x"][event_index],
            arrays["cell_y"][event_index],
            arrays["cell_l"][event_index],
        )
        if not voxels:
            continue
        events_voxels.append(voxels)

    if not events_voxels:
        raise RuntimeError("No usable layer-2 events were found.")

    rows = aggregate_mean_voxels(events_voxels)

    output_png = output_prefix.with_suffix(".png")
    output_csv = output_prefix.with_suffix(".csv")
    output_json = output_prefix.with_suffix(".json")

    save_voxel_table(rows, output_csv)
    plot_voxels(
        rows,
        output_png,
        title=f"Full layer-2 voxel representation in (r, alpha)\n{input_path.name} | events used: {len(events_voxels)}",
    )
    output_json.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "events_requested": int(max_events),
                "events_used": int(len(events_voxels)),
                "num_voxels": int(len(rows)),
                "output_png": str(output_png),
                "output_csv": str(output_csv),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "events_used": len(events_voxels),
                "num_voxels": len(rows),
                "output_png": str(output_png),
                "output_csv": str(output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
