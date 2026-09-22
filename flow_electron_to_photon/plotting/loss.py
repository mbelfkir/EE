from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from matplotlib import pyplot as plt


GRID_ALPHA = 0.25


@dataclass
class GraphDTO:
    values: Iterable[float]
    label: str
    marker: str
    color: str
    markersize: float = 4.5


def plot_loss_batch(
    training_loss: list[float],
    validation_loss: list[float],
    training_l2: list[float],
    validation_l2: list[float],
    learning_rate: list[float],
    batch_loss_prunning: int,
    outpath: str,
) -> None:
    lefts = [
        GraphDTO(values=training_loss, label="Training loss", marker="o", color="#4C78A8"),
        GraphDTO(values=validation_loss, label="Validation loss", marker="o", color="#F58518"),
    ]
    rights = [
        GraphDTO(values=learning_rate, label="Learning rate", marker="s", color="#54A24B"),
    ]
    plot_loss(
        lefts=lefts,
        rights=rights,
        left_ylabel="Loss",
        right_ylabel="Learning rate",
        xlabel=f"Batches × {batch_loss_prunning}",
        title="Batch-level training history",
        outpath=os.path.join(outpath, "loss_batch.png"),
    )


def plot_loss_epoch(
    training_loss: list[float],
    validation_loss: list[float],
    learning_rate: list[float],
    outpath: str,
) -> None:
    lefts = [
        GraphDTO(values=training_loss, label="Training loss", marker="o", color="#4C78A8"),
        GraphDTO(values=validation_loss, label="Validation loss", marker="o", color="#F58518"),
    ]
    rights = [
        GraphDTO(values=learning_rate, label="Learning rate", marker="s", color="#54A24B"),
    ]
    plot_loss(
        lefts=lefts,
        rights=rights,
        left_ylabel="Loss",
        right_ylabel="Learning rate",
        xlabel="Epoch",
        title="Epoch-level training history",
        outpath=os.path.join(outpath, "loss_epoch.png"),
    )


def plot_loss(
    lefts: list[GraphDTO],
    rights: list[GraphDTO],
    left_ylabel: str,
    right_ylabel: str,
    xlabel: str,
    title: str,
    outpath: str,
) -> None:
    fig, left_axis = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    right_axis = left_axis.twinx()

    left_axis.set_xlabel(xlabel)
    left_axis.set_ylabel(left_ylabel)
    left_axis.grid(alpha=GRID_ALPHA)

    for graph in lefts:
        left_axis.plot(
            list(graph.values),
            color=graph.color,
            marker=graph.marker,
            markersize=graph.markersize,
            linewidth=1.7,
            label=graph.label,
        )

    right_axis.set_ylabel(right_ylabel)
    for graph in rights:
        right_axis.plot(
            list(graph.values),
            color=graph.color,
            marker=graph.marker,
            markersize=graph.markersize,
            linewidth=1.5,
            linestyle="--",
            label=graph.label,
        )

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", frameon=False)
    left_axis.set_title(title)

    fig.savefig(outpath, dpi=160)
    plt.close(fig)
