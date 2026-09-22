from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flow.losses import latent_alignment_loss, maximum_mean_discrepancy, reconstruction_loss
from flow.plotting import (
    plot_image_grid,
    plot_latent_histograms,
    plot_latent_statistics,
    plot_latent_wasserstein,
    plot_mean_window_pair,
    plot_mean_window_triplet,
    plot_metric_histograms,
)
from flow.utils import (
    ELECTRON_DOMAIN,
    PHOTON_DOMAIN,
    append_csv_row,
    ensure_dir,
    save_json,
    tensor_batch_to_numpy,
    to_physical_images,
)


CELL_SIZE = 0.025
PHYSICS_TITLES = {
    "reta": "Reta",
    "rphi": "Rphi",
    "weta2": "Weta2",
    "e1x1_over_e7x7": "E1x1 / E7x7",
    "e3x3_over_e7x7": "E3x3 / E7x7",
    "hottest_cell_energy": "Hottest Cell Energy",
    "total_energy": "Total Energy",
    "e7x7": "E7x7",
    "e3x7": "E3x7",
    "e3x3": "E3x3",
    "e1x1": "E1x1",
    "e3x7_over_total": "E3x7 / Total",
    "e3x3_over_total": "E3x3 / Total",
    "e1x1_over_total": "E1x1 / Total",
}


@dataclass
class DomainModelOutputs:
    scaled_inputs: np.ndarray
    physical_inputs: np.ndarray
    latents: np.ndarray
    conditions: np.ndarray
    energy_scales: np.ndarray
    paths: list[str]
    scaled_outputs: np.ndarray | None = None
    physical_outputs: np.ndarray | None = None
    transported_latents: np.ndarray | None = None


def supports_window(windows: np.ndarray, eta_bins: int, phi_bins: int) -> bool:
    return windows.shape[2] >= eta_bins and windows.shape[1] >= phi_bins


def sum_window(window: np.ndarray, eta_bins: int, phi_bins: int) -> float:
    center_eta = window.shape[1] // 2
    center_phi = window.shape[0] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_slice = slice(center_eta - eta_half, center_eta + eta_half + 1)
    phi_slice = slice(center_phi - phi_half, center_phi + phi_half + 1)
    return float(np.asarray(window, dtype=np.float32)[phi_slice, eta_slice].sum())


def compute_weta2(window: np.ndarray, eta_bins: int = 3, phi_bins: int = 5) -> float:
    center_eta = window.shape[1] // 2
    center_phi = window.shape[0] // 2
    eta_half = eta_bins // 2
    phi_half = phi_bins // 2
    eta_slice = slice(center_eta - eta_half, center_eta + eta_half + 1)
    phi_slice = slice(center_phi - phi_half, center_phi + phi_half + 1)
    sub = np.asarray(window, dtype=np.float32)[phi_slice, eta_slice]
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


def compute_physics_features(windows: np.ndarray) -> dict[str, np.ndarray]:
    image_array = np.asarray(windows, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError(f"compute_physics_features expects [N, phi, eta] windows, got {image_array.shape}")
    if not supports_window(image_array, eta_bins=7, phi_bins=7):
        raise ValueError("compute_physics_features requires windows with at least 7 eta bins and 7 phi bins.")

    values: dict[str, list[float]] = {
        "reta": [],
        "rphi": [],
        "weta2": [],
        "e1x1_over_e7x7": [],
        "e3x3_over_e7x7": [],
        "hottest_cell_energy": [],
        "total_energy": [],
        "e7x7": [],
        "e3x7": [],
        "e3x3": [],
        "e1x1": [],
        "e3x7_over_total": [],
        "e3x3_over_total": [],
        "e1x1_over_total": [],
    }

    for window in image_array:
        center_eta = window.shape[1] // 2
        center_phi = window.shape[0] // 2
        total = float(window.sum())
        e7x7 = sum_window(window, 7, 7)
        e3x7 = sum_window(window, 3, 7)
        e3x3 = sum_window(window, 3, 3)
        e1x1 = float(window[center_phi, center_eta])
        reta = 0.0 if e7x7 == 0.0 else e3x7 / e7x7
        rphi = 0.0 if e3x7 == 0.0 else e3x3 / e3x7
        values["reta"].append(reta)
        values["rphi"].append(rphi)
        values["weta2"].append(compute_weta2(window, 3, 5))
        values["e1x1_over_e7x7"].append(0.0 if e7x7 == 0.0 else e1x1 / e7x7)
        values["e3x3_over_e7x7"].append(0.0 if e7x7 == 0.0 else e3x3 / e7x7)
        values["hottest_cell_energy"].append(e1x1)
        values["total_energy"].append(total)
        values["e7x7"].append(e7x7)
        values["e3x7"].append(e3x7)
        values["e3x3"].append(e3x3)
        values["e1x1"].append(e1x1)
        values["e3x7_over_total"].append(0.0 if total == 0.0 else e3x7 / total)
        values["e3x3_over_total"].append(0.0 if total == 0.0 else e3x3 / total)
        values["e1x1_over_total"].append(0.0 if total == 0.0 else e1x1 / total)

    return {key: np.asarray(item, dtype=np.float32) for key, item in values.items()}


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def approx_wasserstein_distance(a: np.ndarray, b: np.ndarray, *, num_quantiles: int = 512) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, num_quantiles)
    return float(np.mean(np.abs(np.quantile(x, quantiles) - np.quantile(y, quantiles))))


def _write_stats_rows(path: Path, group_name: str, features: Mapping[str, np.ndarray]) -> None:
    for metric_name, values in features.items():
        stats = summarize_metric(values)
        append_csv_row(
            path,
            {
                "group": group_name,
                "metric": metric_name,
                **stats,
            },
        )


def _write_comparison_rows(
    path: Path,
    reference_name: str,
    reference_features: Mapping[str, np.ndarray],
    comparison_name: str,
    comparison_features: Mapping[str, np.ndarray],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name, reference_values in reference_features.items():
        comparison_values = comparison_features[metric_name]
        wasserstein = approx_wasserstein_distance(reference_values, comparison_values)
        reference_stats = summarize_metric(reference_values)
        comparison_stats = summarize_metric(comparison_values)
        append_csv_row(
            path,
            {
                "metric": metric_name,
                "reference": reference_name,
                "comparison": comparison_name,
                "wasserstein": wasserstein,
                "reference_mean": reference_stats["mean"],
                "reference_std": reference_stats["std"],
                "comparison_mean": comparison_stats["mean"],
                "comparison_std": comparison_stats["std"],
                "abs_mean_shift": abs(comparison_stats["mean"] - reference_stats["mean"]),
                "abs_std_shift": abs(comparison_stats["std"] - reference_stats["std"]),
            },
        )
        summary[f"{comparison_name}_vs_{reference_name}_{metric_name}_w1"] = wasserstein
    return summary


def latent_comparison_name(comparison: str, reference: str) -> str:
    return f"{comparison}_vs_{reference}"


def summarize_latent_groups(
    latents_by_group: Mapping[str, np.ndarray],
    *,
    reference_label: str | None = None,
) -> dict[str, Any]:
    arrays = {label: np.asarray(values, dtype=np.float32) for label, values in latents_by_group.items()}
    labels = list(arrays.keys())
    if not labels:
        return {"reference": None, "groups": {}, "comparisons": {}}

    resolved_reference = labels[-1] if reference_label is None else str(reference_label)
    if resolved_reference not in arrays:
        raise ValueError(f"Reference latent group '{resolved_reference}' is not available.")

    summary: dict[str, Any] = {
        "reference": resolved_reference,
        "groups": {},
        "comparisons": {},
    }
    for label, values in arrays.items():
        summary["groups"][label] = {
            "mean": values.mean(axis=0).astype(np.float32).tolist(),
            "std": values.std(axis=0).astype(np.float32).tolist(),
        }

    reference_values = arrays[resolved_reference]
    for label, values in arrays.items():
        if label == resolved_reference:
            continue
        per_dimension = np.asarray(
            [
                approx_wasserstein_distance(values[:, dim], reference_values[:, dim])
                for dim in range(values.shape[1])
            ],
            dtype=np.float32,
        )
        top_k = min(5, int(per_dimension.size))
        top_k_values = np.sort(per_dimension)[-top_k:] if top_k > 0 else np.empty((0,), dtype=np.float32)
        comparison_name = latent_comparison_name(label, resolved_reference)
        summary["comparisons"][comparison_name] = {
            "comparison": label,
            "reference": resolved_reference,
            "per_dimension_wasserstein": per_dimension.tolist(),
            "average_wasserstein": float(per_dimension.mean()) if per_dimension.size else 0.0,
            "maximum_wasserstein": float(per_dimension.max()) if per_dimension.size else 0.0,
            "top5_average_wasserstein": float(top_k_values.mean()) if top_k_values.size else 0.0,
        }
    return summary


def _save_latent_statistics(
    latents_by_group: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    reference_label: str | None = None,
) -> dict[str, float]:
    stats_csv = output_dir / "latent_stats.csv"
    comparison_csv = output_dir / "latent_comparisons.csv"
    summary_csv = output_dir / "latent_w1_summary.csv"
    summary = summarize_latent_groups(latents_by_group, reference_label=reference_label)
    metric_summary: dict[str, float] = {}

    for path in (stats_csv, comparison_csv, summary_csv):
        if path.exists():
            path.unlink()

    for label, values in summary["groups"].items():
        mean_values = np.asarray(values["mean"], dtype=np.float32)
        std_values = np.asarray(values["std"], dtype=np.float32)
        for dim in range(mean_values.shape[0]):
            append_csv_row(
                stats_csv,
                {
                    "group": label,
                    "latent_dim": dim,
                    "mean": float(mean_values[dim]),
                    "std": float(std_values[dim]),
                },
            )

    wasserstein_curves: dict[str, np.ndarray] = {}
    for comparison_name, values in summary["comparisons"].items():
        per_dimension = np.asarray(values["per_dimension_wasserstein"], dtype=np.float32)
        wasserstein_curves[comparison_name] = per_dimension
        for dim, distance in enumerate(per_dimension):
            append_csv_row(
                comparison_csv,
                {
                    "comparison": str(values["comparison"]),
                    "reference": str(values["reference"]),
                    "latent_dim": dim,
                    "wasserstein": float(distance),
                },
            )
        append_csv_row(
            summary_csv,
            {
                "comparison": str(values["comparison"]),
                "reference": str(values["reference"]),
                "average_wasserstein": float(values["average_wasserstein"]),
                "maximum_wasserstein": float(values["maximum_wasserstein"]),
                "top5_average_wasserstein": float(values["top5_average_wasserstein"]),
            },
        )
        metric_summary[f"{comparison_name}_average_latent_w1"] = float(values["average_wasserstein"])
        metric_summary[f"{comparison_name}_maximum_latent_w1"] = float(values["maximum_wasserstein"])
        metric_summary[f"{comparison_name}_top5_average_latent_w1"] = float(values["top5_average_wasserstein"])

    save_json(summary, output_dir / "latent_summary.json")
    plot_latent_wasserstein(wasserstein_curves, output_dir / "latent_wasserstein.png")
    return metric_summary


@torch.no_grad()
def collect_encoder_latents(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    encoder_was_training = encoder.training
    encoder.eval()

    latents: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        latent = encoder(images)
        latents.append(latent.detach().cpu().numpy().astype(np.float32))

    if encoder_was_training:
        encoder.train()

    if not latents:
        raise ValueError("collect_encoder_latents received an empty loader.")
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def evaluate_latent_alignment(
    encoder: nn.Module,
    electron_loader: DataLoader,
    photon_loader: DataLoader,
    device: torch.device,
    *,
    alignment_loss_name: str = "mmd",
) -> dict[str, float]:
    electron_latents = collect_encoder_latents(encoder, electron_loader, device)
    photon_latents = collect_encoder_latents(encoder, photon_loader, device)
    summary = summarize_latent_groups(
        {
            ELECTRON_DOMAIN: electron_latents,
            PHOTON_DOMAIN: photon_latents,
        },
        reference_label=PHOTON_DOMAIN,
    )
    comparison_name = latent_comparison_name(ELECTRON_DOMAIN, PHOTON_DOMAIN)
    comparison = summary["comparisons"][comparison_name]
    alignment_value = latent_alignment_loss(
        torch.from_numpy(electron_latents).to(device),
        torch.from_numpy(photon_latents).to(device),
        mode=alignment_loss_name,
    )
    return {
        "latent_align_loss": float(alignment_value.item()),
        "avg_latent_w1": float(comparison["average_wasserstein"]),
        "max_latent_w1": float(comparison["maximum_wasserstein"]),
        "top5_avg_latent_w1": float(comparison["top5_average_wasserstein"]),
        f"{comparison_name}_average_latent_w1": float(comparison["average_wasserstein"]),
        f"{comparison_name}_maximum_latent_w1": float(comparison["maximum_wasserstein"]),
        f"{comparison_name}_top5_average_latent_w1": float(comparison["top5_average_wasserstein"]),
    }


@torch.no_grad()
def evaluate_autoencoder_loss(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    l1_weight: float = 1.0,
    l2_weight: float = 0.0,
) -> dict[str, float]:
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    total = 0.0
    l1_total = 0.0
    l2_total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        reconstructions = decoder(encoder(images))
        loss, pieces = reconstruction_loss(reconstructions, images, l1_weight=l1_weight, l2_weight=l2_weight)
        batch_size = images.size(0)
        total += float(loss.item()) * batch_size
        l1_total += float(pieces["l1"].item()) * batch_size
        l2_total += float(pieces["l2"].item()) * batch_size
        count += batch_size

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()

    denominator = max(count, 1)
    return {
        "loss": total / denominator,
        "l1": l1_total / denominator,
        "l2": l2_total / denominator,
        "count": float(count),
    }


@torch.no_grad()
def collect_autoencoder_outputs(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> DomainModelOutputs:
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    encoder.eval()
    decoder.eval()

    scaled_inputs: list[np.ndarray] = []
    scaled_outputs: list[np.ndarray] = []
    physical_inputs: list[np.ndarray] = []
    physical_outputs: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    energy_scales: list[np.ndarray] = []
    paths: list[str] = []

    for batch in loader:
        images = batch["image"].to(device)
        latent = encoder(images)
        reconstructed = decoder(latent)
        image_np = tensor_batch_to_numpy(images)
        reconstructed_np = tensor_batch_to_numpy(reconstructed)
        energy_np = batch["energy_scale"].detach().cpu().numpy().astype(np.float32).reshape(-1)

        scaled_inputs.append(image_np)
        scaled_outputs.append(reconstructed_np)
        physical_inputs.append(to_physical_images(image_np, energy_np))
        physical_outputs.append(to_physical_images(reconstructed_np, energy_np))
        latents.append(latent.detach().cpu().numpy().astype(np.float32))
        conditions.append(batch["cond"].detach().cpu().numpy().astype(np.float32))
        energy_scales.append(energy_np)
        paths.extend(list(batch["path"]))

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()

    return DomainModelOutputs(
        scaled_inputs=np.concatenate(scaled_inputs, axis=0),
        scaled_outputs=np.concatenate(scaled_outputs, axis=0),
        physical_inputs=np.concatenate(physical_inputs, axis=0),
        physical_outputs=np.concatenate(physical_outputs, axis=0),
        latents=np.concatenate(latents, axis=0),
        conditions=np.concatenate(conditions, axis=0),
        energy_scales=np.concatenate(energy_scales, axis=0),
        paths=paths,
    )


@torch.no_grad()
def collect_flow_translation_outputs(
    encoder: nn.Module,
    decoder: nn.Module,
    flow: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    source_domain: int,
    target_domain: int,
) -> DomainModelOutputs:
    encoder_was_training = encoder.training
    decoder_was_training = decoder.training
    flow_was_training = flow.training
    encoder.eval()
    decoder.eval()
    flow.eval()

    scaled_inputs: list[np.ndarray] = []
    scaled_outputs: list[np.ndarray] = []
    physical_inputs: list[np.ndarray] = []
    physical_outputs: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    transported_latents: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    energy_scales: list[np.ndarray] = []
    paths: list[str] = []

    for batch in loader:
        images = batch["image"].to(device)
        latent = encoder(images)
        transported = flow.transport(latent, source_domain=source_domain, target_domain=target_domain)
        translated = decoder(transported)

        image_np = tensor_batch_to_numpy(images)
        translated_np = tensor_batch_to_numpy(translated)
        energy_np = batch["energy_scale"].detach().cpu().numpy().astype(np.float32).reshape(-1)

        scaled_inputs.append(image_np)
        scaled_outputs.append(translated_np)
        physical_inputs.append(to_physical_images(image_np, energy_np))
        physical_outputs.append(to_physical_images(translated_np, energy_np))
        latents.append(latent.detach().cpu().numpy().astype(np.float32))
        transported_latents.append(transported.detach().cpu().numpy().astype(np.float32))
        conditions.append(batch["cond"].detach().cpu().numpy().astype(np.float32))
        energy_scales.append(energy_np)
        paths.extend(list(batch["path"]))

    if encoder_was_training:
        encoder.train()
    if decoder_was_training:
        decoder.train()
    if flow_was_training:
        flow.train()

    return DomainModelOutputs(
        scaled_inputs=np.concatenate(scaled_inputs, axis=0),
        scaled_outputs=np.concatenate(scaled_outputs, axis=0),
        physical_inputs=np.concatenate(physical_inputs, axis=0),
        physical_outputs=np.concatenate(physical_outputs, axis=0),
        latents=np.concatenate(latents, axis=0),
        transported_latents=np.concatenate(transported_latents, axis=0),
        conditions=np.concatenate(conditions, axis=0),
        energy_scales=np.concatenate(energy_scales, axis=0),
        paths=paths,
    )


def _save_domain_reconstruction_artifacts(
    domain_name: str,
    outputs: DomainModelOutputs,
    output_dir: Path,
    *,
    max_plot_items: int = 6,
) -> dict[str, float]:
    if outputs.physical_outputs is None:
        raise ValueError("Reconstruction artifacts require output images.")

    domain_dir = ensure_dir(output_dir / domain_name)
    original_features = compute_physics_features(outputs.physical_inputs)
    reconstructed_features = compute_physics_features(outputs.physical_outputs)

    plot_image_grid(
        {"Original": outputs.physical_inputs, "Reconstructed": outputs.physical_outputs},
        domain_dir / "examples.png",
        max_items=max_plot_items,
        title=f"{domain_name.capitalize()} validation samples",
    )
    plot_mean_window_pair(
        outputs.physical_inputs,
        outputs.physical_outputs,
        domain_dir / "mean_windows.png",
        reference_label=f"{domain_name.capitalize()} original",
        candidate_label=f"{domain_name.capitalize()} reconstructed",
    )
    plot_metric_histograms(
        {"original": original_features, "reconstructed": reconstructed_features},
        domain_dir / "histograms",
        PHYSICS_TITLES,
    )

    stats_csv = domain_dir / "feature_stats.csv"
    comparisons_csv = domain_dir / "feature_comparisons.csv"
    _write_stats_rows(stats_csv, "original", original_features)
    _write_stats_rows(stats_csv, "reconstructed", reconstructed_features)
    summary = _write_comparison_rows(
        comparisons_csv,
        "original",
        original_features,
        "reconstructed",
        reconstructed_features,
    )
    summary.update(
        {
            f"{domain_name}_scaled_l1": float(np.mean(np.abs(outputs.scaled_outputs - outputs.scaled_inputs))),
            f"{domain_name}_scaled_l2": float(np.mean((outputs.scaled_outputs - outputs.scaled_inputs) ** 2)),
        }
    )
    save_json(
        {
            "original": {key: summarize_metric(values) for key, values in original_features.items()},
            "reconstructed": {key: summarize_metric(values) for key, values in reconstructed_features.items()},
            "comparisons": summary,
        },
        domain_dir / "summary.json",
    )
    return summary


def run_autoencoder_validation(
    encoder: nn.Module,
    decoder: nn.Module,
    electron_loader: DataLoader,
    photon_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    *,
    max_plot_items: int = 6,
) -> dict[str, float]:
    output_dir = ensure_dir(output_dir)
    electron_outputs = collect_autoencoder_outputs(encoder, decoder, electron_loader, device)
    photon_outputs = collect_autoencoder_outputs(encoder, decoder, photon_loader, device)

    metrics: dict[str, float] = {}
    metrics.update(
        _save_domain_reconstruction_artifacts(
            ELECTRON_DOMAIN,
            electron_outputs,
            output_dir,
            max_plot_items=max_plot_items,
        )
    )
    metrics.update(
        _save_domain_reconstruction_artifacts(
            PHOTON_DOMAIN,
            photon_outputs,
            output_dir,
            max_plot_items=max_plot_items,
        )
    )

    latent_dir = ensure_dir(output_dir / "latent")
    latent_groups = {
        ELECTRON_DOMAIN: electron_outputs.latents,
        PHOTON_DOMAIN: photon_outputs.latents,
    }
    plot_latent_histograms(latent_groups, latent_dir / "latent_histograms.png")
    plot_latent_statistics(latent_groups, latent_dir / "latent_statistics.png")
    metrics.update(_save_latent_statistics(latent_groups, latent_dir, reference_label=PHOTON_DOMAIN))

    metrics["combined_scaled_l1"] = float(
        0.5 * (
            np.mean(np.abs(electron_outputs.scaled_outputs - electron_outputs.scaled_inputs))
            + np.mean(np.abs(photon_outputs.scaled_outputs - photon_outputs.scaled_inputs))
        )
    )
    metrics["combined_scaled_l2"] = float(
        0.5 * (
            np.mean((electron_outputs.scaled_outputs - electron_outputs.scaled_inputs) ** 2)
            + np.mean((photon_outputs.scaled_outputs - photon_outputs.scaled_inputs) ** 2)
        )
    )
    metrics["avg_latent_w1"] = float(metrics[f"{ELECTRON_DOMAIN}_vs_{PHOTON_DOMAIN}_average_latent_w1"])
    metrics["max_latent_w1"] = float(metrics[f"{ELECTRON_DOMAIN}_vs_{PHOTON_DOMAIN}_maximum_latent_w1"])
    metrics["top5_avg_latent_w1"] = float(metrics[f"{ELECTRON_DOMAIN}_vs_{PHOTON_DOMAIN}_top5_average_latent_w1"])
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def run_flow_validation(
    encoder: nn.Module,
    decoder: nn.Module,
    flow: nn.Module,
    electron_loader: DataLoader,
    photon_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    *,
    max_plot_items: int = 6,
) -> dict[str, float]:
    output_dir = ensure_dir(output_dir)
    electron_outputs = collect_flow_translation_outputs(
        encoder,
        decoder,
        flow,
        electron_loader,
        device,
        source_domain=0,
        target_domain=1,
    )
    photon_outputs = collect_autoencoder_outputs(encoder, decoder, photon_loader, device)

    if electron_outputs.physical_outputs is None or electron_outputs.transported_latents is None:
        raise ValueError("Flow validation requires transported outputs and transported latents.")

    plot_image_grid(
        {
            "Real electron": electron_outputs.physical_inputs,
            "Translated photon": electron_outputs.physical_outputs,
            "Real photon": photon_outputs.physical_inputs,
        },
        output_dir / "translated_examples.png",
        max_items=max_plot_items,
        title="Electron to photon latent transport",
    )
    plot_mean_window_triplet(
        electron_outputs.physical_inputs,
        electron_outputs.physical_outputs,
        photon_outputs.physical_inputs,
        output_dir / "mean_windows.png",
        source_label="Electron mean",
        translated_label="Translated photon mean",
        target_label="Photon mean",
    )

    electron_features = compute_physics_features(electron_outputs.physical_inputs)
    translated_features = compute_physics_features(electron_outputs.physical_outputs)
    photon_features = compute_physics_features(photon_outputs.physical_inputs)
    plot_metric_histograms(
        {
            "electron": electron_features,
            "translated_photon": translated_features,
            "photon": photon_features,
        },
        output_dir / "histograms",
        PHYSICS_TITLES,
    )

    stats_csv = output_dir / "feature_stats.csv"
    comparisons_csv = output_dir / "feature_comparisons.csv"
    _write_stats_rows(stats_csv, "electron", electron_features)
    _write_stats_rows(stats_csv, "translated_photon", translated_features)
    _write_stats_rows(stats_csv, "photon", photon_features)

    metrics: dict[str, float] = {}
    metrics.update(_write_comparison_rows(comparisons_csv, "photon", photon_features, "electron", electron_features))
    metrics.update(
        _write_comparison_rows(comparisons_csv, "photon", photon_features, "translated_photon", translated_features)
    )

    latent_dir = ensure_dir(output_dir / "latent")
    latent_groups = {
        "electron_latent": electron_outputs.latents,
        "transported_electron_latent": electron_outputs.transported_latents,
        "photon_latent": photon_outputs.latents,
    }
    plot_latent_histograms(latent_groups, latent_dir / "latent_histograms.png")
    plot_latent_statistics(latent_groups, latent_dir / "latent_statistics.png")
    metrics.update(_save_latent_statistics(latent_groups, latent_dir, reference_label="photon_latent"))

    electron_latent_tensor = torch.from_numpy(electron_outputs.latents).to(device)
    transported_latent_tensor = torch.from_numpy(electron_outputs.transported_latents).to(device)
    photon_latent_tensor = torch.from_numpy(photon_outputs.latents).to(device)
    with torch.no_grad():
        metrics["electron_nll"] = float((-flow.log_prob(electron_latent_tensor, 0).mean()).item())
        metrics["photon_nll"] = float((-flow.log_prob(photon_latent_tensor, 1).mean()).item())
        metrics["transported_nll_in_photon_domain"] = float((-flow.log_prob(transported_latent_tensor, 1).mean()).item())
        metrics["latent_transport_mmd"] = float(maximum_mean_discrepancy(transported_latent_tensor, photon_latent_tensor).item())

    primary_keys = [
        "translated_photon_vs_photon_total_energy_w1",
        "translated_photon_vs_photon_reta_w1",
        "translated_photon_vs_photon_rphi_w1",
        "translated_photon_vs_photon_weta2_w1",
        "translated_photon_vs_photon_e1x1_over_e7x7_w1",
        "translated_photon_vs_photon_e3x3_over_e7x7_w1",
    ]
    metrics["selection_score"] = float(sum(metrics[key] for key in primary_keys if key in metrics))

    save_json(
        {
            "features": {
                "electron": {key: summarize_metric(values) for key, values in electron_features.items()},
                "translated_photon": {key: summarize_metric(values) for key, values in translated_features.items()},
                "photon": {key: summarize_metric(values) for key, values in photon_features.items()},
            },
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    return metrics
