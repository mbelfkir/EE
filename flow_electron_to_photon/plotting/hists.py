from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import hist
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

Binning = Tuple[int, float, float]

COLORS = {
    "target": "#54A24B",
    "proto": "#4C78A8",
    "proto_corr": "#F58518",
    "proto_fudged": "#B279A2",
    "bkg": "#E45756",
    "neutral": "#222222",
}
GRID_ALPHA = 0.25


def _normalized_weights(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0:
        return array
    total = float(np.sum(array))
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.full(array.shape, 1.0 / len(array), dtype=np.float64)
    return array / total


def _hist_arrays(histogram: hist.BaseHist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = histogram.values()
    variances = histogram.variances()
    if variances is None:
        variances = np.zeros_like(values)
    centers = histogram.axes[0].centers
    return centers, values, np.sqrt(np.clip(variances, 0.0, None))


def _plot_histogram(
    axis: plt.Axes,
    histogram: hist.BaseHist,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.9,
) -> None:
    centers, values, errors = _hist_arrays(histogram)
    axis.step(centers, values, where="mid", label=label, color=color, linewidth=linewidth, linestyle=linestyle)
    axis.fill_between(
        centers,
        np.clip(values - errors, 0.0, None),
        values + errors,
        step="mid",
        color=color,
        alpha=0.14,
    )


def _compute_ratio(
    numerator_values: np.ndarray,
    numerator_errors: np.ndarray,
    denominator_values: np.ndarray,
    denominator_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.full_like(numerator_values, np.nan, dtype=np.float64)
    error = np.full_like(numerator_values, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (np.abs(numerator_values) > 0.0)
        & (np.abs(denominator_values) > 0.0)
    )
    ratio[valid] = numerator_values[valid] / denominator_values[valid]
    error[valid] = np.abs(ratio[valid]) * np.sqrt(
        (numerator_errors[valid] / numerator_values[valid]) ** 2
        + (denominator_errors[valid] / denominator_values[valid]) ** 2
    )
    return ratio, error


def _plot_ratio(axis: plt.Axes, centers: np.ndarray, ratio: np.ndarray, error: np.ndarray, *, color: str, label: str | None = None) -> None:
    axis.step(centers, ratio, where="mid", color=color, linewidth=1.7, label=label)
    axis.fill_between(
        centers,
        ratio - error,
        ratio + error,
        step="mid",
        color=color,
        alpha=0.14,
    )


def _set_panel_style(axis: plt.Axes, *, ylabel: str, show_xlabel: bool = False, xlabel: str = "") -> None:
    axis.set_ylabel(ylabel)
    axis.grid(alpha=GRID_ALPHA)
    if show_xlabel:
        axis.set_xlabel(xlabel)
    else:
        axis.set_xlabel("")


def _build_title(var_label: str, annotation: str, subannotation: str, hint: str = "") -> str:
    lines = [var_label]
    if annotation:
        lines.append(annotation)
    if subannotation:
        lines.append(subannotation)
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def plot_bases(
    hist_proto: np.ndarray,
    hist_target: np.ndarray,
    centers: np.ndarray,
    score: float,
    var: str,
    outpath: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    axis.step(centers, hist_proto, where="mid", label="Source latent", color=COLORS["proto"], linewidth=1.8)
    axis.step(centers, hist_target, where="mid", label="Target latent", color=COLORS["target"], linewidth=1.8)
    axis.set_title(f"{var} latent comparison\nL2 = {score ** 0.5:.2e}")
    axis.set_xlabel(var)
    axis.set_ylabel("Density")
    axis.grid(alpha=GRID_ALPHA)
    axis.legend(frameon=False)
    fig.savefig(os.path.join(outpath, f"{var}_base.png"), dpi=160)
    plt.close(fig)


def plot_input_hists(
    proto: pd.DataFrame,
    target: pd.DataFrame,
    variables: Iterable[str],
    weight_var: str,
    variable_labels: Dict[str, str],
    binning: Dict[str, Binning],
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    outpath: str,
) -> None:
    for var in variables:
        hist_proto = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_target = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_proto.fill(proto[var], weight=_normalized_weights(proto[weight_var]))
        hist_target.fill(target[var], weight=_normalized_weights(target[weight_var]))
        plot_hists(
            hist_proto=hist_proto,
            hist_target=hist_target,
            var=var,
            var_label=variable_labels.get(var, var),
            binning=binning[var],
            annotation=annotation,
            subannotation=subannotation,
            ylabel=ylabel,
            ratio_ylabel=ratio_ylabel,
            legend_labels=legend_labels,
            outpath=outpath,
        )


def plot_hists(
    hist_proto: hist.BaseHist,
    hist_target: hist.BaseHist,
    var: str,
    var_label: str,
    binning: Binning,
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    outpath: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        sharex=True,
    )
    top_axis, ratio_axis = axes

    _plot_histogram(
        top_axis,
        hist_target,
        label=legend_labels.get("target", "Target"),
        color=COLORS["target"],
    )
    _plot_histogram(
        top_axis,
        hist_proto,
        label=legend_labels.get("proto", "Source"),
        color=COLORS["proto"],
    )

    centers_target, target_values, target_errors = _hist_arrays(hist_target)
    _, proto_values, proto_errors = _hist_arrays(hist_proto)
    ratio, ratio_error = _compute_ratio(target_values, target_errors, proto_values, proto_errors)
    _plot_ratio(ratio_axis, centers_target, ratio, ratio_error, color=COLORS["neutral"])

    ymax = max(float(np.nanmax(target_values)) if target_values.size else 0.0, float(np.nanmax(proto_values)) if proto_values.size else 0.0)
    top_axis.set_ylim(0.0, 1.18 * ymax if ymax > 0 else 1.0)
    top_axis.set_xlim(binning[1], binning[2])
    top_axis.set_title(_build_title(var_label, annotation, subannotation))
    top_axis.legend(frameon=False)
    _set_panel_style(top_axis, ylabel=ylabel)

    ratio_axis.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=1.0)
    ratio_axis.set_ylim(0.5, 1.5)
    _set_panel_style(ratio_axis, ylabel=ratio_ylabel, show_xlabel=True, xlabel=var_label)

    print(f"\033[1;36m[INFO]\033[92m 👍  Plot for {var} is done! \033[0m")
    fig.savefig(os.path.join(outpath, f"{var}.pdf"), dpi=160)
    plt.close(fig)


def plot_corrected_hists(
    proto: pd.DataFrame,
    target: pd.DataFrame,
    variables: Iterable[str],
    weight_var: str,
    variable_labels: Dict[str, str],
    binning: Dict[str, Binning],
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    corr_suffix: str,
    outpath: str,
    hint: str = "",
    filename_suffix: str = "",
) -> None:
    for var in variables:
        hist_proto = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_proto_corr = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_target = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()

        weight_proto = _normalized_weights(proto[weight_var])
        hist_proto.fill(proto[var], weight=weight_proto)
        hist_proto_corr.fill(proto[f"{var}{corr_suffix}"], weight=weight_proto)
        hist_target.fill(target[var], weight=_normalized_weights(target[weight_var]))

        plot_hists_comparison(
            hists_proto={"proto": hist_proto, "proto_corr": hist_proto_corr},
            hist_target=hist_target,
            var=var,
            var_label=variable_labels.get(var, var),
            binning=binning[var],
            annotation=annotation,
            subannotation=subannotation,
            ylabel=ylabel,
            ratio_ylabel=ratio_ylabel,
            legend_labels=legend_labels,
            colors={
                "target": COLORS["target"],
                "proto": COLORS["proto"],
                "proto_corr": COLORS["proto_corr"],
                "proto_fudged": COLORS["proto_fudged"],
            },
            outpath=outpath,
            hint=hint,
            filename_suffix=filename_suffix,
        )


def plot_corrected_fudged_hists(
    proto: pd.DataFrame,
    target: pd.DataFrame,
    variables: Iterable[str],
    weight_var: str,
    variable_labels: Dict[str, str],
    binning: Dict[str, Binning],
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    corr_suffix: str,
    fudged_prefix: str,
    outpath: str,
    hint: str = "",
) -> None:
    for var in variables:
        hist_proto = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_proto_corr = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_proto_fudged = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_target = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()

        weight_proto = _normalized_weights(proto[weight_var])
        hist_proto.fill(proto[var], weight=weight_proto)
        hist_proto_corr.fill(proto[f"{var}{corr_suffix}"], weight=weight_proto)
        hist_proto_fudged.fill(proto[var.replace(fudged_prefix, "")], weight=weight_proto)
        hist_target.fill(target[var], weight=_normalized_weights(target[weight_var]))

        plot_hists_comparison(
            hists_proto={
                "proto": hist_proto,
                "proto_corr": hist_proto_corr,
                "proto_fudged": hist_proto_fudged,
            },
            hist_target=hist_target,
            var=var,
            var_label=variable_labels.get(var, var),
            binning=binning[var],
            annotation=annotation,
            subannotation=subannotation,
            ylabel=ylabel,
            ratio_ylabel=ratio_ylabel,
            legend_labels=legend_labels,
            colors={
                "target": COLORS["target"],
                "proto": COLORS["proto"],
                "proto_corr": COLORS["proto_corr"],
                "proto_fudged": COLORS["proto_fudged"],
            },
            outpath=outpath,
            hint=hint,
        )


def plot_hists_comparison(
    hists_proto: Dict[str, hist.BaseHist],
    hist_target: hist.BaseHist,
    var: str,
    var_label: str,
    binning: Binning,
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    colors: Dict[str, str],
    outpath: str,
    hint: str = "",
    filename_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.8, 8.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.8, 1.3, 1.0]},
        sharex=True,
    )
    top_axis, ratio_axis, zoom_axis = axes

    _plot_histogram(
        top_axis,
        hist_target,
        label=legend_labels.get("target", "Target"),
        color=colors.get("target", COLORS["target"]),
    )

    target_centers, target_values, target_errors = _hist_arrays(hist_target)
    ymax = float(np.nanmax(target_values)) if target_values.size else 0.0
    for name, histogram in hists_proto.items():
        _plot_histogram(
            top_axis,
            histogram,
            label=legend_labels.get(name, name),
            color=colors.get(name, COLORS["neutral"]),
        )
        _, values, _ = _hist_arrays(histogram)
        ymax = max(ymax, float(np.nanmax(values)) if values.size else 0.0)

    top_axis.set_ylim(0.0, 1.18 * ymax if ymax > 0 else 1.0)
    top_axis.set_xlim(binning[1], binning[2])
    top_axis.set_title(_build_title(var_label, annotation, subannotation, hint))
    top_axis.legend(frameon=False)
    _set_panel_style(top_axis, ylabel=ylabel)

    ratio_axis.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=1.0)
    zoom_axis.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=1.0)

    for name, histogram in hists_proto.items():
        _, values, errors = _hist_arrays(histogram)
        ratio, error = _compute_ratio(target_values, target_errors, values, errors)
        color = colors.get(name, COLORS["neutral"])
        _plot_ratio(ratio_axis, target_centers, ratio, error, color=color, label=legend_labels.get(name, name))
        _plot_ratio(zoom_axis, target_centers, ratio, error, color=color)

    ratio_axis.set_ylim(0.0, 2.0)
    zoom_axis.set_ylim(0.8, 1.2)
    ratio_axis.legend(frameon=False, ncols=min(len(hists_proto), 3))
    _set_panel_style(ratio_axis, ylabel=ratio_ylabel)
    _set_panel_style(zoom_axis, ylabel=ratio_ylabel, show_xlabel=True, xlabel=var_label)

    if filename_suffix:
        filename = f"{var}_{filename_suffix}.pdf"
    else:
        filename = f"{var}.pdf"
    print(f"\033[1;36m[INFO]\033[92m 👍  Plot {filename} corrected is done! \033[0m")
    fig.savefig(os.path.join(outpath, filename), dpi=160)
    plt.close(fig)


def plot_target_sub_bkg_corrected_hists(
    proto_corr: pd.DataFrame,
    target_only: pd.DataFrame,
    bkg: pd.DataFrame,
    variables: Iterable[str],
    weight_var: str,
    variable_labels: Dict[str, str],
    binning: Dict[str, Binning],
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    corr_suffix: str,
    outpath: str,
    hint: str = "",
) -> None:
    for var in variables:
        hist_proto_corr = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_target_only = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_bkg = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()
        hist_total = hist.new.Reg(*binning[var], overflow=False, underflow=False).Weight()

        hist_proto_corr.fill(
            proto_corr[f"{var}{corr_suffix}"],
            weight=_normalized_weights(proto_corr[weight_var]),
        )
        if len(target_only) > 0:
            hist_target_only.fill(target_only[var], weight=_normalized_weights(target_only[weight_var]))
        if len(bkg) > 0:
            hist_bkg.fill(bkg[var], weight=_normalized_weights(bkg[weight_var]))
        hist_total.view().value = hist_proto_corr.values() + hist_bkg.values()
        hist_total.view().variance = np.nan_to_num(hist_proto_corr.variances(), nan=0.0) + np.nan_to_num(hist_bkg.variances(), nan=0.0)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.8, 6.8),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3.2, 1.2]},
            sharex=True,
        )
        top_axis, ratio_axis = axes

        _plot_histogram(
            top_axis,
            hist_target_only,
            label=legend_labels.get("target", "Target"),
            color=COLORS["target"],
        )
        _plot_histogram(
            top_axis,
            hist_total,
            label=f"{legend_labels.get('proto_corr', 'Corrected')} + {legend_labels.get('bkg', 'Background')}",
            color=COLORS["proto_corr"],
        )
        _plot_histogram(
            top_axis,
            hist_bkg,
            label=legend_labels.get("bkg", "Background"),
            color=COLORS["bkg"],
            linestyle="--",
            linewidth=1.5,
        )

        centers, target_values, target_errors = _hist_arrays(hist_target_only)
        _, total_values, total_errors = _hist_arrays(hist_total)
        ratio, error = _compute_ratio(target_values, target_errors, total_values, total_errors)
        _plot_ratio(ratio_axis, centers, ratio, error, color=COLORS["neutral"])

        top_axis.set_xlim(binning[var][1], binning[var][2])
        top_axis.set_title(_build_title(variable_labels[var], annotation, subannotation, hint))
        top_axis.legend(frameon=False)
        _set_panel_style(top_axis, ylabel=ylabel)

        ratio_axis.axhline(1.0, color=COLORS["neutral"], linestyle="--", linewidth=1.0)
        ratio_axis.set_ylim(0.8, 1.2)
        _set_panel_style(ratio_axis, ylabel=ratio_ylabel, show_xlabel=True, xlabel=variable_labels[var])

        print(f"\033[1;36m[INFO]\033[92m 👍  Plot for {var} corrected is done! \033[0m")
        fig.savefig(os.path.join(outpath, f"{var}.pdf"), dpi=160)
        plt.close(fig)
