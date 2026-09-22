from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    l1_weight: float = 1.0,
    l2_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Primary reconstruction objective with an optional L2 hook."""

    l1 = F.l1_loss(prediction, target)
    l2 = F.mse_loss(prediction, target)
    total = l1_weight * l1 + l2_weight * l2
    return total, {
        "l1": l1.detach(),
        "l2": l2.detach(),
        "total": total.detach(),
    }


def pairwise_squared_distance(x: Tensor, y: Tensor) -> Tensor:
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
    return (x_norm + y_norm - 2.0 * x @ y.t()).clamp_min(0.0)


def _validate_latent_pair(x: Tensor, y: Tensor, loss_name: str) -> None:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"{loss_name} expects [batch, dim] tensors, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.size(1) != y.size(1):
        raise ValueError(f"{loss_name} requires matching feature dimensions, got {x.size(1)} and {y.size(1)}.")


def maximum_mean_discrepancy(
    x: Tensor,
    y: Tensor,
    kernel_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    eps: float = 1e-6,
) -> Tensor:
    """Stable multi-kernel RBF MMD for latent-domain alignment."""

    _validate_latent_pair(x, y, "MMD")

    combined = torch.cat([x, y], dim=0)
    distances = pairwise_squared_distance(combined, combined)
    positive = distances[distances > 0.0]
    if positive.numel() == 0:
        base_bandwidth = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    else:
        base_bandwidth = positive.median().detach().clamp_min(eps)

    xx = pairwise_squared_distance(x, x)
    yy = pairwise_squared_distance(y, y)
    xy = pairwise_squared_distance(x, y)

    kernel_xx = torch.zeros_like(xx)
    kernel_yy = torch.zeros_like(yy)
    kernel_xy = torch.zeros_like(xy)
    for multiplier in kernel_multipliers:
        bandwidth = base_bandwidth * float(multiplier)
        kernel_xx = kernel_xx + torch.exp(-xx / bandwidth)
        kernel_yy = kernel_yy + torch.exp(-yy / bandwidth)
        kernel_xy = kernel_xy + torch.exp(-xy / bandwidth)

    return kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()


def coral_loss(
    source: Tensor,
    target: Tensor,
    *,
    align_mean: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """CORAL loss with an optional mean-alignment term for latent vectors."""

    _validate_latent_pair(source, target, "CORAL")

    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_denom = max(source.size(0) - 1, 1)
    target_denom = max(target.size(0) - 1, 1)
    source_cov = (source_centered.t() @ source_centered) / float(source_denom)
    target_cov = (target_centered.t() @ target_centered) / float(target_denom)

    feature_dim = max(source.size(1), 1)
    covariance_term = ((source_cov - target_cov) ** 2).sum() / (4.0 * float(feature_dim * feature_dim))
    if not align_mean:
        return covariance_term

    mean_term = F.mse_loss(source_mean, target_mean)
    return covariance_term + mean_term


def latent_alignment_loss(source: Tensor, target: Tensor, mode: str = "mmd") -> Tensor:
    """Dispatch between the supported Stage-1 latent alignment objectives."""

    normalized_mode = str(mode).lower()
    if normalized_mode == "mmd":
        return maximum_mean_discrepancy(source, target)
    if normalized_mode == "coral":
        return coral_loss(source, target)
    raise ValueError(f"Unsupported latent alignment loss '{mode}'.")


def latent_moment_loss(source: Tensor, target: Tensor) -> Tensor:
    """Simple moment-matching penalty for latent transport batches."""

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("latent_moment_loss expects [batch, dim] tensors.")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_std = source.std(dim=0, unbiased=False)
    target_std = target.std(dim=0, unbiased=False)
    return F.l1_loss(source_mean, target_mean) + F.l1_loss(source_std, target_std)
