from __future__ import annotations

import os
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np


def plot_corr_matrices_combined(
    matrix: np.ndarray,
    matrix_corrected: np.ndarray,
    variables: list[str],
    variable_labels: Dict[str, str],
    outpath: str,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    filename = f"corr_matrix_{filename_suffix}.png"
    title = f"Correlation matrix {title_suffix}".strip()

    labels = [variable_labels[var] for var in variables]
    corrected_labels = [f"{label}'" for label in labels]

    matrix_joined = matrix.copy()
    lower = np.tril_indices_from(matrix_joined, k=-1)
    matrix_joined[lower] = matrix_corrected[lower]

    plot_corr_matrices(
        matrix=matrix_joined,
        xlabels=labels,
        ylabels=corrected_labels,
        title=title,
        outpath=outpath,
        filename=filename,
    )


def plot_corr_matrices(
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    outpath: str,
    filename: str,
) -> None:
    size = matrix.shape[0]
    fig, axis = plt.subplots(
        figsize=(max(7.5, 1.05 * size + 3.0), max(6.0, 0.95 * size + 2.5)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    axis.set_xticks(np.arange(size))
    axis.set_yticks(np.arange(size))
    axis.set_xticklabels(xlabels, rotation=45, ha="right")
    axis.set_yticklabels(ylabels)
    axis.set_title(title)

    axis.set_xticks(np.arange(-0.5, size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0, alpha=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(size):
        for column in range(size):
            value = float(matrix[row, column])
            text_color = "white" if abs(value) > 0.55 else "#222222"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(image, ax=axis, shrink=0.86, label="Weighted correlation")
    fig.savefig(os.path.join(outpath, filename), dpi=160)
    plt.close(fig)
