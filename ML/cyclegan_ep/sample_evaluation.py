from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "ML/cyclegan_ep/sample_evaluation.py requires matplotlib, numpy, and torch."
    ) from exc

from data import restore_physical_array
from representation_utils import stack_window_representations
from utils import normalized_to_raw_numpy, raw_to_normalized_numpy


CELL_SIZE = 0.025


def supports_window(windows: np.ndarray, eta_bins: int, phi_bins: int) -> bool:
    return windows.shape[2] >= eta_bins and windows.shape[1] >= phi_bins


def sum_window(window: np.ndarray, eta_bins: int, phi_bins: int) -> float:
    center_eta = window.shape[1] // 2
    center_phi = window.shape[0] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_start = center_eta - eta_half
    eta_stop = center_eta + eta_half + 1
    phi_start = center_phi - phi_half
    phi_stop = center_phi + phi_half + 1
    return float(window[phi_start:phi_stop, eta_start:eta_stop].sum())


def compute_weta2(window: np.ndarray, eta_bins: int = 3, phi_bins: int = 5) -> float:
    center_eta = window.shape[1] // 2
    center_phi = window.shape[0] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_slice = slice(center_eta - eta_half, center_eta + eta_half + 1)
    phi_slice = slice(center_phi - phi_half, center_phi + phi_half + 1)
    sub = window[phi_slice, eta_slice]
    total = float(sub.sum())
    if total == 0.0:
        return 0.0
    eta_indices = np.arange(sub.shape[1], dtype=np.float32)
    eta_offsets = (eta_indices - eta_half) * CELL_SIZE
    eta_grid = np.tile(eta_offsets[None, :], (sub.shape[0], 1))
    mean_eta = float((sub * eta_grid).sum() / total)
    mean_eta2 = float((sub * (eta_grid ** 2)).sum() / total)
    variance = max(0.0, mean_eta2 - mean_eta ** 2)
    return variance ** 0.5


def compute_shapes(windows: np.ndarray) -> Dict[str, np.ndarray]:
    if not supports_window(windows, eta_bins=7, phi_bins=7):
        raise ValueError("compute_shapes requires windows with at least 7 eta bins and 7 phi bins.")

    reta = []
    rphi = []
    weta2 = []
    e1x1_over_e7x7 = []
    e3x3_over_e7x7 = []
    hottest_cell_energy = []

    for window in windows:
        center_eta = window.shape[1] // 2
        center_phi = window.shape[0] // 2
        e7x7 = sum_window(window, 7, 7)
        e3x7 = sum_window(window, 3, 7)
        e3x3 = sum_window(window, 3, 3)
        e1x1 = float(window[center_phi, center_eta])

        reta.append(0.0 if e7x7 == 0.0 else e3x7 / e7x7)
        rphi.append(0.0 if e3x7 == 0.0 else e3x3 / e3x7)
        weta2.append(compute_weta2(window, 3, 5))
        e1x1_over_e7x7.append(0.0 if e7x7 == 0.0 else e1x1 / e7x7)
        e3x3_over_e7x7.append(0.0 if e7x7 == 0.0 else e3x3 / e7x7)
        hottest_cell_energy.append(e1x1)

    return {
        "reta": np.asarray(reta, dtype=np.float32),
        "rphi": np.asarray(rphi, dtype=np.float32),
        "weta2": np.asarray(weta2, dtype=np.float32),
        "e1x1_over_e7x7": np.asarray(e1x1_over_e7x7, dtype=np.float32),
        "e3x3_over_e7x7": np.asarray(e3x3_over_e7x7, dtype=np.float32),
        "hottest_cell_energy": np.asarray(hottest_cell_energy, dtype=np.float32),
    }


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def translate_windows_in_batches(
    generator: nn.Module,
    windows: np.ndarray,
    conditions: np.ndarray | None,
    energy_scales: np.ndarray | None,
    device: torch.device,
    normalization_scale: float,
    batch_size: int,
    symmetric_range: bool,
    model_backend: str,
    phi_window_size: int,
    eta_window_size: int,
) -> np.ndarray:
    if windows.size == 0:
        return np.empty((0, 0, 0), dtype=np.float32)

    generator_was_training = generator.training
    generator.eval()
    translated_batches = []
    with torch.no_grad():
        if model_backend == "image":
            for start in range(0, len(windows), batch_size):
                stop = min(start + batch_size, len(windows))
                batch = raw_to_normalized_numpy(
                    windows[start:stop],
                    normalization_scale,
                    symmetric_range=symmetric_range,
                )
                batch_tensor = torch.from_numpy(batch[:, None, :, :]).to(device)
                translated = generator(batch_tensor)
                translated_batches.append(
                    np.stack(
                        [
                            normalized_to_raw_numpy(item, normalization_scale, symmetric_range=symmetric_range)
                            for item in translated
                        ],
                        axis=0,
                    )
                )
                if device.type == "mps":
                    torch.mps.synchronize()
        elif model_backend == "voxel":
            if conditions is None or energy_scales is None:
                raise ValueError("Voxel evaluation requires condition values and beam-energy scales.")
            voxels = stack_window_representations(windows, representation="voxel")
            voxels[:, :, 0] /= energy_scales
            for start in range(0, len(windows), batch_size):
                stop = min(start + batch_size, len(windows))
                voxel_tensor = torch.from_numpy(voxels[start:stop]).to(device)
                condition_tensor = torch.from_numpy(conditions[start:stop]).to(device)
                translated_energy = generator(voxel_tensor, condition_tensor)
                translated_batches.append(
                    (
                        translated_energy.detach().cpu().numpy().astype(np.float32)
                        * energy_scales[start:stop]
                    ).reshape(-1, phi_window_size, eta_window_size)
                )
                if device.type == "mps":
                    torch.mps.synchronize()
        else:
            raise ValueError(f"Unsupported model_backend: {model_backend}")

    if generator_was_training:
        generator.train()

    return np.concatenate(translated_batches, axis=0)


def plot_mean_windows(
    electron_windows: np.ndarray,
    translated_windows: np.ndarray,
    photon_windows: np.ndarray,
    output_path: Path,
) -> None:
    mean_e = electron_windows.mean(axis=0)
    mean_t = translated_windows.mean(axis=0)
    mean_p = photon_windows.mean(axis=0)
    diff = mean_t - mean_p

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    images = [
        (mean_e, "Electron mean"),
        (mean_t, "Translated photon mean"),
        (mean_p, "Real photon mean"),
        (diff, "Translated - real"),
    ]
    for ax, (image, title) in zip(axes, images):
        im = ax.imshow(image, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("eta bin")
        ax.set_ylabel("phi bin")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_total_energy_comparison(
    electron_totals: np.ndarray,
    translated_totals: np.ndarray,
    photon_totals: np.ndarray,
    output_path: Path,
) -> None:
    combined = np.concatenate([electron_totals, translated_totals, photon_totals])
    xmin = float(np.percentile(combined, 0.5))
    xmax = float(np.percentile(combined, 99.5))
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    bins = np.linspace(xmin, xmax, 60)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(electron_totals, bins=bins, density=True, histtype="step", linewidth=1.8, label="electron")
    ax.hist(translated_totals, bins=bins, density=True, histtype="step", linewidth=1.8, label="translated photon")
    ax.hist(photon_totals, bins=bins, density=True, histtype="step", linewidth=1.8, label="real photon")
    ax.set_xlabel("Window energy")
    ax.set_ylabel("Density")
    ax.set_title("Total Energy Comparison")
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_core_3x5_windows(
    electron_windows: np.ndarray,
    translated_windows: np.ndarray,
    photon_windows: np.ndarray,
    output_path: Path,
) -> Dict[str, float]:
    center_eta = electron_windows.shape[2] // 2
    center_phi = electron_windows.shape[1] // 2
    eta_slice = slice(center_eta - 1, center_eta + 2)
    phi_slice = slice(center_phi - 2, center_phi + 3)

    mean_e = electron_windows.mean(axis=0)[phi_slice, eta_slice]
    mean_t = translated_windows.mean(axis=0)[phi_slice, eta_slice]
    mean_p = photon_windows.mean(axis=0)[phi_slice, eta_slice]
    diff_t = mean_t - mean_p
    diff_e = mean_e - mean_p
    ratio_t = np.divide(mean_t, mean_p, out=np.full_like(mean_t, np.nan), where=mean_p != 0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (mean_e, "Electron mean 3x5", "viridis"),
        (mean_t, "Translated mean 3x5", "viridis"),
        (mean_p, "Photon mean 3x5", "viridis"),
        (diff_e, "Electron - photon", "coolwarm"),
        (diff_t, "Translated - photon", "coolwarm"),
        (ratio_t, "Translated / photon", "magma"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("eta offset bin")
        ax.set_ylabel("phi offset bin")
        for iy in range(image.shape[0]):
            for ix in range(image.shape[1]):
                value = image[iy, ix]
                if np.isfinite(value):
                    ax.text(ix, iy, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    return {
        "electron_vs_photon_core_mse": float(np.mean((mean_e - mean_p) ** 2)),
        "translated_vs_photon_core_mse": float(np.mean((mean_t - mean_p) ** 2)),
        "electron_vs_photon_core_mae": float(np.mean(np.abs(mean_e - mean_p))),
        "translated_vs_photon_core_mae": float(np.mean(np.abs(mean_t - mean_p))),
    }


def plot_window_comparison(
    electron_windows: np.ndarray,
    translated_windows: np.ndarray,
    photon_windows: np.ndarray,
    output_path: Path,
    eta_bins: int,
    phi_bins: int,
    tag: str,
) -> Dict[str, float]:
    center_eta = electron_windows.shape[2] // 2
    center_phi = electron_windows.shape[1] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_slice = slice(center_eta - eta_half, center_eta + eta_half + 1)
    phi_slice = slice(center_phi - phi_half, center_phi + phi_half + 1)

    mean_e = electron_windows.mean(axis=0)[phi_slice, eta_slice]
    mean_t = translated_windows.mean(axis=0)[phi_slice, eta_slice]
    mean_p = photon_windows.mean(axis=0)[phi_slice, eta_slice]
    diff_t = mean_t - mean_p
    diff_e = mean_e - mean_p
    ratio_t = np.divide(mean_t, mean_p, out=np.full_like(mean_t, np.nan), where=mean_p != 0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = [
        (mean_e, f"Electron mean {tag}", "viridis"),
        (mean_t, f"Translated mean {tag}", "viridis"),
        (mean_p, f"Photon mean {tag}", "viridis"),
        (diff_e, "Electron - photon", "coolwarm"),
        (diff_t, "Translated - photon", "coolwarm"),
        (ratio_t, "Translated / photon", "magma"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("eta offset bin")
        ax.set_ylabel("phi offset bin")
        for iy in range(image.shape[0]):
            for ix in range(image.shape[1]):
                value = image[iy, ix]
                if np.isfinite(value):
                    ax.text(ix, iy, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)

    return {
        f"electron_vs_photon_{tag}_mse": float(np.mean((mean_e - mean_p) ** 2)),
        f"translated_vs_photon_{tag}_mse": float(np.mean((mean_t - mean_p) ** 2)),
        f"electron_vs_photon_{tag}_mae": float(np.mean(np.abs(mean_e - mean_p))),
        f"translated_vs_photon_{tag}_mae": float(np.mean(np.abs(mean_t - mean_p))),
    }


def plot_shape_histograms(
    electron_shapes: Dict[str, np.ndarray],
    translated_shapes: Dict[str, np.ndarray],
    photon_shapes: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    configs = {
        "reta": "Reta",
        "rphi": "Rphi",
        "weta2": "Weta2",
        "e1x1_over_e7x7": "E1x1 / E7x7",
        "e3x3_over_e7x7": "E3x3 / E7x7",
        "hottest_cell_energy": "Hottest Cell Energy",
    }
    for key, title in configs.items():
        electron = electron_shapes[key]
        translated = translated_shapes[key]
        photon = photon_shapes[key]

        combined = np.concatenate([electron, translated, photon])
        xmin = float(np.percentile(combined, 0.5))
        xmax = float(np.percentile(combined, 99.5))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        pad = 0.08 * (xmax - xmin)
        bins = np.linspace(xmin - pad, xmax + pad, 41)
        centers = 0.5 * (bins[1:] + bins[:-1])
        width = bins[1] - bins[0]

        def density_and_error(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            counts, _ = np.histogram(values, bins=bins)
            norm = max(len(values) * width, 1e-12)
            density = counts / norm
            error = np.sqrt(counts) / norm
            return density, error

        electron_density, electron_error = density_and_error(electron)
        translated_density, translated_error = density_and_error(translated)
        photon_density, photon_error = density_and_error(photon)

        fig, (ax, ratio_ax) = plt.subplots(
            2,
            1,
            figsize=(7, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )

        for density, error, label in [
            (electron_density, electron_error, "electron"),
            (translated_density, translated_error, "translated photon"),
            (photon_density, photon_error, "real photon"),
        ]:
            ax.stairs(density, bins, linewidth=1.8, label=label)
            ax.bar(
                centers,
                2.0 * error,
                bottom=np.clip(density - error, 0.0, None),
                width=width,
                alpha=0.15,
                align="center",
            )

        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend()

        def ratio_and_error(num: np.ndarray, num_err: np.ndarray, den: np.ndarray, den_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            ratio = np.full_like(num, np.nan, dtype=np.float64)
            error = np.full_like(num, np.nan, dtype=np.float64)
            mask = den > 0
            ratio[mask] = num[mask] / den[mask]
            safe_num = np.where(num > 0, num, np.nan)
            rel_num = np.where(np.isfinite(safe_num), num_err / safe_num, 0.0)
            rel_den = np.zeros_like(den_err)
            rel_den[mask] = den_err[mask] / den[mask]
            error[mask] = ratio[mask] * np.sqrt(rel_num[mask] ** 2 + rel_den[mask] ** 2)
            return ratio, error

        ele_ratio, ele_ratio_err = ratio_and_error(electron_density, electron_error, photon_density, photon_error)
        tr_ratio, tr_ratio_err = ratio_and_error(translated_density, translated_error, photon_density, photon_error)

        ratio_ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ratio_ax.stairs(ele_ratio, bins, linewidth=1.4, label="electron / photon")
        ratio_ax.bar(
            centers,
            2.0 * np.nan_to_num(ele_ratio_err, nan=0.0),
            bottom=np.nan_to_num(ele_ratio - ele_ratio_err, nan=0.0),
            width=width,
            alpha=0.15,
            align="center",
        )
        ratio_ax.stairs(tr_ratio, bins, linewidth=1.4, label="translated / photon")
        ratio_ax.bar(
            centers,
            2.0 * np.nan_to_num(tr_ratio_err, nan=0.0),
            bottom=np.nan_to_num(tr_ratio - tr_ratio_err, nan=0.0),
            width=width,
            alpha=0.15,
            align="center",
        )
        ratio_ax.set_ylabel("Ratio")
        ratio_ax.set_xlabel(title)
        ratio_ax.set_ylim(0.5, 1.5 if key != "hottest_cell_energy" else 2.0)

        fig.savefig(output_dir / f"{key}_comparison.png", dpi=160)
        plt.close(fig)


def run_sample_evaluation(
    generator_ab: nn.Module,
    model_backend: str,
    electron_windows: np.ndarray,
    electron_conditions: np.ndarray | None,
    electron_energy_scales: np.ndarray | None,
    photon_windows: np.ndarray,
    photon_conditions: np.ndarray | None,
    photon_energy_scales: np.ndarray | None,
    sample_dir: Path,
    device: torch.device,
    normalization_scale: float,
    batch_size: int,
    preprocessing: Dict[str, object] | None,
    standardization: Dict[str, object] | None,
    symmetric_range: bool,
    phi_window_size: int,
    eta_window_size: int,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    translated = translate_windows_in_batches(
        generator=generator_ab,
        windows=electron_windows,
        conditions=electron_conditions,
        energy_scales=electron_energy_scales,
        device=device,
        normalization_scale=normalization_scale,
        batch_size=batch_size,
        symmetric_range=symmetric_range,
        model_backend=model_backend,
        phi_window_size=phi_window_size,
        eta_window_size=eta_window_size,
    )

    electron_windows = restore_physical_array(electron_windows, preprocessing=preprocessing, standardization=standardization)
    photon_windows = restore_physical_array(photon_windows, preprocessing=preprocessing, standardization=standardization)
    translated = restore_physical_array(translated, preprocessing=preprocessing, standardization=standardization)

    electron_totals = electron_windows.sum(axis=(1, 2)).astype(np.float32)
    photon_totals = photon_windows.sum(axis=(1, 2)).astype(np.float32)
    translated_totals = translated.sum(axis=(1, 2)).astype(np.float32)

    plot_mean_windows(electron_windows, translated, photon_windows, sample_dir / "mean_windows_comparison.png")
    plot_total_energy_comparison(electron_totals, translated_totals, photon_totals, sample_dir / "total_energy_comparison.png")

    if supports_window(electron_windows, eta_bins=3, phi_bins=5):
        core_metrics = plot_core_3x5_windows(
            electron_windows,
            translated,
            photon_windows,
            sample_dir / "core_3x5_window_comparison.png",
        )
    else:
        core_metrics = {"skipped": True, "reason": "Window is smaller than 3x5."}

    if supports_window(electron_windows, eta_bins=7, phi_bins=11):
        window_7x11_metrics = plot_window_comparison(
            electron_windows,
            translated,
            photon_windows,
            sample_dir / "window_7x11_comparison.png",
            eta_bins=7,
            phi_bins=11,
            tag="7x11",
        )
        electron_shapes = compute_shapes(electron_windows)
        translated_shapes = compute_shapes(translated)
        photon_shapes = compute_shapes(photon_windows)
        plot_shape_histograms(electron_shapes, translated_shapes, photon_shapes, sample_dir)
        shower_shapes = {
            "electron": {key: summarize_metric(val) for key, val in electron_shapes.items()},
            "translated": {key: summarize_metric(val) for key, val in translated_shapes.items()},
            "photon": {key: summarize_metric(val) for key, val in photon_shapes.items()},
        }
    else:
        window_7x11_metrics = {"skipped": True, "reason": "Window is smaller than 7x11."}
        shower_shapes = {"skipped": True, "reason": "Window is smaller than 7x11."}

    metrics = {
        "num_electron_eval": int(len(electron_windows)),
        "num_photon_eval": int(len(photon_windows)),
        "mean_window_mse_translated_vs_real": float(np.mean((translated.mean(axis=0) - photon_windows.mean(axis=0)) ** 2)),
        "mean_window_mse_electron_vs_real": float(np.mean((electron_windows.mean(axis=0) - photon_windows.mean(axis=0)) ** 2)),
        "core_3x5_metrics": core_metrics,
        "window_7x11_metrics": window_7x11_metrics,
        "total_energy": {
            "electron": summarize_metric(electron_totals),
            "translated": summarize_metric(translated_totals),
            "photon": summarize_metric(photon_totals),
        },
        "shower_shapes": shower_shapes,
    }
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
