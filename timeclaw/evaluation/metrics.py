"""Numerical metrics for TimeClaw benchmark evaluation.

All functions are pure and accept python floats / lists / numpy arrays. They
return NaN (not raise) when undefined so per-task failures don't poison the
aggregate.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-8


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------

def _as_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric MAPE in [0, 200]. Returns NaN on length mismatch."""
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp) + EPS
    return float(100.0 * np.mean(2.0 * np.abs(yt - yp) / denom))


def maape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Arctangent Absolute Percentage Error in [0, π/2].

    Robust to zero values where MAPE explodes. See Kim & Kim, 2016.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    return float(np.mean(np.arctan(np.abs((yt - yp) / (yt + EPS)))))


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


# ---------------------------------------------------------------------------
# CRPS / RCRPS (CiK)
# ---------------------------------------------------------------------------

def crps_from_quantiles(
    quantile_levels: Sequence[float],
    quantile_values: Sequence[float],
    y_true: float,
) -> float:
    """Continuous Ranked Probability Score from a quantile forecast.

    Uses the trapezoidal integral of (F̂(z) - 𝟙{z ≥ y})² over z. The
    quantile-level / quantile-value pairs must be sorted by quantile level.
    For point forecasts (single quantile), this reduces to |y - ŷ|.
    """
    if len(quantile_levels) != len(quantile_values) or len(quantile_levels) == 0:
        return float("nan")
    if len(quantile_levels) == 1:
        return float(abs(float(y_true) - float(quantile_values[0])))

    qs = np.asarray(quantile_levels, dtype=float)
    vs = np.asarray(quantile_values, dtype=float)
    order = np.argsort(qs)
    qs, vs = qs[order], vs[order]
    y = float(y_true)

    # Build (z, F̂(z)) by interleaving y and the quantile values.
    # F̂(z) for z in [v_i, v_{i+1}] equals q_i (right-continuous step).
    # The squared difference (F̂(z) - 1{z>=y})^2 piecewise-constant per
    # segment, integrated analytically per segment.
    zs = np.concatenate(([min(vs[0], y) - 1e-6], vs, [max(vs[-1], y) + 1e-6]))
    fs = np.concatenate(([0.0], qs, [1.0]))
    # left-aligned: F̂ on [z_i, z_{i+1}] is fs[i]
    total = 0.0
    for i in range(len(zs) - 1):
        a, b = zs[i], zs[i + 1]
        f = fs[i]
        if b <= y:
            indicator = 0.0
            total += (f - indicator) ** 2 * (b - a)
        elif a >= y:
            indicator = 1.0
            total += (f - indicator) ** 2 * (b - a)
        else:
            # y inside segment, split
            total += (f - 0.0) ** 2 * (y - a)
            total += (f - 1.0) ** 2 * (b - y)
    return float(total)


def rcrps(
    y_true: Sequence[float],
    forecast: Sequence[float] | dict,
    region_of_interest: Sequence[int] | None,
    metric_scaling: float = 1.0,
) -> float:
    """Region-of-Interest CRPS (CiK paper).

    ``forecast`` is either a 1D sequence of point predictions (one per step,
    aligned with ``y_true``) or a dict ``{"levels": [...], "samples": [[...]]}``
    where ``samples[t]`` are quantile values at step t for ``levels``.
    """
    yt = _as_array(y_true)
    if yt.size == 0:
        return float("nan")
    if region_of_interest is None or len(region_of_interest) == 0:
        roi = np.arange(yt.size)
    else:
        roi = np.asarray([i for i in region_of_interest if 0 <= i < yt.size], dtype=int)
        if roi.size == 0:
            return float("nan")

    crps_per_t: list[float] = []
    if isinstance(forecast, dict):
        levels = forecast.get("levels") or []
        samples = forecast.get("samples") or []
        if len(samples) != yt.size:
            return float("nan")
        for t in roi:
            crps_per_t.append(
                crps_from_quantiles(levels, samples[t], float(yt[t]))
            )
    else:
        yp = _as_array(forecast)
        if yp.shape != yt.shape:
            return float("nan")
        for t in roi:
            crps_per_t.append(float(abs(float(yt[t]) - float(yp[t]))))

    return float(np.mean(crps_per_t) * float(metric_scaling))


# ---------------------------------------------------------------------------
# Classification / MCQ
# ---------------------------------------------------------------------------

def accuracy(preds: Iterable, golds: Iterable) -> float:
    """Exact-match accuracy. NaN if empty / length mismatch."""
    p = list(preds)
    g = list(golds)
    if not p or len(p) != len(g):
        return float("nan")
    return float(sum(1 for a, b in zip(p, g) if a is not None and a == b) / len(p))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_numeric(
    values: Sequence[float], how: str = "mean"
) -> float:
    """Reduce a list of per-task metrics, ignoring NaN."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    if how == "mean":
        return float(arr.mean())
    if how == "median":
        return float(np.median(arr))
    if how == "sum":
        return float(arr.sum())
    raise ValueError(f"unknown aggregation: {how}")
