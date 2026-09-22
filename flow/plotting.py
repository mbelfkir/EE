from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _combined_range(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    flattened = np.concatenate([np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays if np.asarray(array).size > 0])
    if flattened.size == 0:
        return 0.0, 1.0
    minimum = float(np.min(flattened))
    maximum = float(np.max(flattened))
    if minimum == maximum:
        maximum = minimum + 1.0
    return minimum, maximum


def plot_training_history(
    rows: Sequence[Mapping[str, float | int]],
    output_path: str | Path,
    keys: Sequence[tuple[str, str]],
    title: str,
) -> None:
    if not rows:
        return

    output_path = _ensure_parent(output_path)
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, max(3.0 * len(keys), 3.5)), constrained_layout=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, keys):
        values = [float(row.get(key, np.nan)) for row in rows]
        ax.plot(epochs, values, marker="o", linewidth=1.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_grid(
    image_groups: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    max_items: int = 6,
    title: str | None = None,
) -> None:
    if not image_groups:
        return
    arrays = {label: np.asarray(images, dtype=np.float32) for label, images in image_groups.items()}
    if any(array.ndim != 3 for array in arrays.values()):
        raise ValueError("plot_image_grid expects each image group to have shape [N, phi, eta].")

    count = min(max_items, min(array.shape[0] for array in arrays.values()))
    if count <= 0:
        return

    output_path = _ensure_parent(output_path)
    labels = list(arrays.keys())
    vmin, vmax = _combined_range([array[:count] for array in arrays.values()])
    fig, axes = plt.subplots(len(labels), count, figsize=(3.2 * count, 3.0 * len(labels)), constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(len(labels), count)

    for row_index, label in enumerate(labels):
        for column_index in range(count):
            axis = axes[row_index, column_index]
            image = arrays[label][column_index]
            im = axis.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            if row_index == 0:
                axis.set_title(f"Sample {column_index + 1}")
            if column_index == 0:
                axis.set_ylabel(f"{label}\nphi")
            axis.set_xlabel("eta")
        fig.colorbar(im, ax=list(axes[row_index]), shrink=0.8)

    if title:
        fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_mean_window_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    output_path: str | Path,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    reference_mean = np.asarray(reference, dtype=np.float32).mean(axis=0)
    candidate_mean = np.asarray(candidate, dtype=np.float32).mean(axis=0)
    difference = candidate_mean - reference_mean
    vmin, vmax = _combined_range([reference_mean, candidate_mean])
    output_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    panels = [
        (reference_mean, reference_label, "viridis", vmin, vmax),
        (candidate_mean, candidate_label, "viridis", vmin, vmax),
        (difference, f"{candidate_label} - {reference_label}", "coolwarm", None, None),
    ]
    for axis, (image, panel_title, cmap, panel_vmin, panel_vmax) in zip(axes, panels):
        im = axis.imshow(image, origin="lower", cmap=cmap, vmin=panel_vmin, vmax=panel_vmax)
        axis.set_title(panel_title)
        axis.set_xlabel("eta")
        axis.set_ylabel("phi")
        fig.colorbar(im, ax=axis, shrink=0.82)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_mean_window_triplet(
    source: np.ndarray,
    translated: np.ndarray,
    target: np.ndarray,
    output_path: str | Path,
    *,
    source_label: str,
    translated_label: str,
    target_label: str,
) -> None:
    source_mean = np.asarray(source, dtype=np.float32).mean(axis=0)
    translated_mean = np.asarray(translated, dtype=np.float32).mean(axis=0)
    target_mean = np.asarray(target, dtype=np.float32).mean(axis=0)
    difference = translated_mean - target_mean
    vmin, vmax = _combined_range([source_mean, translated_mean, target_mean])
    output_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    panels = [
        (source_mean, source_label, "viridis", vmin, vmax),
        (translated_mean, translated_label, "viridis", vmin, vmax),
        (target_mean, target_label, "viridis", vmin, vmax),
        (difference, f"{translated_label} - {target_label}", "coolwarm", None, None),
    ]
    for axis, (image, panel_title, cmap, panel_vmin, panel_vmax) in zip(axes, panels):
        im = axis.imshow(image, origin="lower", cmap=cmap, vmin=panel_vmin, vmax=panel_vmax)
        axis.set_title(panel_title)
        axis.set_xlabel("eta")
        axis.set_ylabel("phi")
        fig.colorbar(im, ax=axis, shrink=0.82)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_histogram_groups(
    groups: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    arrays = [np.asarray(values, dtype=np.float32).reshape(-1) for values in groups.values() if np.asarray(values).size > 0]
    if not arrays:
        return
    combined = np.concatenate(arrays)
    xmin = float(np.percentile(combined, 0.5))
    xmax = float(np.percentile(combined, 99.5))
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    histogram_bins = np.linspace(xmin, xmax, bins)

    output_path = _ensure_parent(output_path)
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for label, values in groups.items():
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        axis.hist(array, bins=histogram_bins, density=True, histtype="step", linewidth=1.8, label=label)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_histograms(
    metric_groups: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str | Path,
    title_map: Mapping[str, str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not metric_groups:
        return

    metrics = sorted(next(iter(metric_groups.values())).keys())
    for metric in metrics:
        groups = {
            label: np.asarray(values[metric], dtype=np.float32)
            for label, values in metric_groups.items()
            if metric in values
        }
        if groups:
            plot_histogram_groups(
                groups,
                output_dir / f"{metric}.png",
                title=title_map.get(metric, metric),
                xlabel=title_map.get(metric, metric),
            )


def plot_latent_histograms(
    latents_by_group: Mapping[str, np.ndarray],
    output_path: str | Path,
    *,
    max_cols: int = 4,
) -> None:
    if not latents_by_group:
        return
    arrays = {label: np.asarray(values, dtype=np.float32) for label, values in latents_by_group.items()}
    latent_dim = next(iter(arrays.values())).shape[1]
    n_cols = min(max_cols, latent_dim)
    n_rows = int(np.ceil(latent_dim / n_cols))
    output_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for dim in range(latent_dim):
        axis = axes[dim // n_cols, dim % n_cols]
        combined = np.concatenate([values[:, dim] for values in arrays.values()], axis=0)
        xmin = float(np.percentile(combined, 0.5))
        xmax = float(np.percentile(combined, 99.5))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        bins = np.linspace(xmin, xmax, 40)
        for label, values in arrays.items():
            axis.hist(values[:, dim], bins=bins, density=True, histtype="step", linewidth=1.6, label=label)
        axis.set_title(f"Latent dim {dim}")
        axis.set_xlabel("Value")
        axis.set_ylabel("Density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    for dim in range(latent_dim, n_rows * n_cols):
        axes[dim // n_cols, dim % n_cols].axis("off")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_statistics(latents_by_group: Mapping[str, np.ndarray], output_path: str | Path) -> None:
    if not latents_by_group:
        return
    arrays = {label: np.asarray(values, dtype=np.float32) for label, values in latents_by_group.items()}
    latent_dim = next(iter(arrays.values())).shape[1]
    output_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    dims = np.arange(latent_dim)
    for label, values in arrays.items():
        axes[0].plot(dims, values.mean(axis=0), marker="o", linewidth=1.4, label=label)
        axes[1].plot(dims, values.std(axis=0), marker="o", linewidth=1.4, label=label)
    axes[0].set_title("Latent mean per dimension")
    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("Mean")
    axes[0].legend()
    axes[1].set_title("Latent std per dimension")
    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel("Std")
    axes[1].legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_wasserstein(comparison_groups: Mapping[str, np.ndarray], output_path: str | Path) -> None:
    if not comparison_groups:
        return

    arrays = {
        label: np.asarray(values, dtype=np.float32).reshape(-1)
        for label, values in comparison_groups.items()
        if np.asarray(values).size > 0
    }
    if not arrays:
        return

    latent_dim = next(iter(arrays.values())).shape[0]
    dims = np.arange(latent_dim)
    output_path = _ensure_parent(output_path)

    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for label, values in arrays.items():
        axis.plot(dims, values, marker="o", linewidth=1.6, label=label)
    axis.set_title("Latent Wasserstein-1 distance per dimension")
    axis.set_xlabel("Latent dimension")
    axis.set_ylabel("Wasserstein-1")
    axis.grid(alpha=0.25)
    if len(arrays) > 1:
        axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
