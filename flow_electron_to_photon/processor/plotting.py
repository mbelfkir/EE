from __future__ import annotations

import os
from typing import Dict

import pandas as pd

from flow_electron_to_photon.misc.correlations import weighted_correlation
from flow_electron_to_photon.misc.system import setup_output_dir
from flow_electron_to_photon.plotting.correlations import plot_corr_matrices_combined
from flow_electron_to_photon.plotting.hists import Binning, plot_corrected_hists


def plot_correlation_matrices(
    proto: pd.DataFrame,
    target: pd.DataFrame,
    variables: list[str],
    weight_var: str,
    corr_suffix: str,
    variable_labels: Dict[str, str],
    outpath: str,
) -> None:
    shower_shapes_corr = [f"{var}{corr_suffix}" for var in variables]
    correlation_matrix_target = weighted_correlation(target[variables], target[weight_var].to_numpy())
    correlation_matrix_proto = weighted_correlation(proto[variables], proto[weight_var].to_numpy())
    correlation_matrix_proto_corr = weighted_correlation(proto[shower_shapes_corr], proto[weight_var].to_numpy())

    plot_corr_matrices_combined(
        matrix=correlation_matrix_target - correlation_matrix_proto,
        matrix_corrected=correlation_matrix_target - correlation_matrix_proto_corr,
        variables=variables,
        variable_labels=variable_labels,
        outpath=outpath,
        title_suffix="Target - Source",
        filename_suffix="target_sub_source",
    )
    plot_corr_matrices_combined(
        matrix=correlation_matrix_target / correlation_matrix_proto,
        matrix_corrected=correlation_matrix_target / correlation_matrix_proto_corr,
        variables=variables,
        variable_labels=variable_labels,
        outpath=outpath,
        title_suffix="Target / Source",
        filename_suffix="target_div_source",
    )
    plot_corr_matrices_combined(
        matrix=correlation_matrix_proto,
        matrix_corrected=correlation_matrix_proto_corr,
        variables=variables,
        variable_labels=variable_labels,
        outpath=outpath,
        title_suffix="Source",
        filename_suffix="source",
    )
    plot_corr_matrices_combined(
        matrix=correlation_matrix_target,
        matrix_corrected=correlation_matrix_target,
        variables=variables,
        variable_labels=variable_labels,
        outpath=outpath,
        title_suffix="Target",
        filename_suffix="target",
    )


def plot_corrected_hists_slices(
    proto: pd.DataFrame,
    target: pd.DataFrame,
    variables: list[str],
    weight_var: str,
    corr_suffix: str,
    slices: Dict[str, list[float]],
    variable_labels: Dict[str, str],
    binning: Dict[str, Binning],
    annotation: str,
    subannotation: str,
    ylabel: str,
    ratio_ylabel: str,
    legend_labels: Dict[str, str],
    outpath: str,
    hint: str = "",
) -> None:
    for cut_var, cut_borders in slices.items():
        outpath_slices_var = setup_output_dir(os.path.join(outpath, cut_var))
        for index, cut_left in enumerate(cut_borders[:-1]):
            cut_right = cut_borders[index + 1]
            label = f"{cut_left} <= {variable_labels[cut_var]} <= {cut_right}"
            filename_suffix = f"{cut_var}_{index}"
            full_hint = label if hint == "" else f"{hint}, {label}"
            plot_corrected_hists(
                proto=proto[(proto[cut_var] >= cut_left) & (proto[cut_var] <= cut_right)],
                target=target[(target[cut_var] >= cut_left) & (target[cut_var] <= cut_right)],
                variables=variables,
                weight_var=weight_var,
                variable_labels=variable_labels,
                binning=binning,
                annotation=annotation,
                subannotation=subannotation,
                ylabel=ylabel,
                ratio_ylabel=ratio_ylabel,
                legend_labels=legend_labels,
                corr_suffix=corr_suffix,
                outpath=outpath_slices_var,
                hint=full_hint,
                filename_suffix=filename_suffix,
            )
