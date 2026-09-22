from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_correlation(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    values = df.to_numpy(dtype=float)
    weights = np.asarray(w, dtype=float).reshape(-1)
    total_weight = weights.sum()
    if not np.isfinite(total_weight) or abs(total_weight) < 1e-9:
        raise ValueError("Sum of weights is approximately zero.")

    mean = (weights[:, None] * values).sum(axis=0) / total_weight
    centered = values - mean
    covariance = (weights[:, None] * centered).T @ centered / total_weight

    variances = np.diag(covariance)
    denom = np.sqrt(variances[:, None] * variances[None, :])
    corr = covariance / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr


def weighted_ranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted, w_sorted = x[order], w[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    ranks = cum_w / w.sum()
    out = np.empty_like(ranks)
    out[order] = ranks
    return out


def weighted_spearman(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    values = df.to_numpy(dtype=float)
    weights = np.asarray(w, dtype=float).reshape(-1)
    ranks = np.column_stack([weighted_ranks(values[:, idx], weights) for idx in range(values.shape[1])])
    return _weighted_correlation(pd.DataFrame(ranks, columns=df.columns), weights)


weighted_correlation = weighted_spearman
