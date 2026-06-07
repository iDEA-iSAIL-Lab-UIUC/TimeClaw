"""Deterministic ~20-dim numerical fingerprint for time-series memory retrieval.

The extractor takes the same ``{channels, timestamps?, meta?}`` payload that
``timeclaw.tools.server.load_data`` consumes, so the same series object can be
fed into both the MCP server (for agent reasoning) and the fingerprint
function (for memory keying) without re-normalization.

Design choices worth keeping in mind:

- Output dim is **fixed at FINGERPRINT_DIM regardless of input shape**
  (univariate, multivariate, short, irregular). Missing features are filled
  with 0.0 so the FAISS index has a consistent geometry.
- For multivariate series we reduce per-channel features to a single vector
  by **averaging across channels** (level/shape stats) plus a small structural
  block (channel count, mean pairwise corr). This loses per-channel detail,
  but a 20-dim retrieval key isn't trying to be sufficient — just selective.
- No RNG, no LLM, pure numpy + scipy. The same series always yields the same
  fingerprint, which is required for retrieval keys.
- All channel values are float-coerced; NaN/inf are dropped before stats so
  the fingerprint stays finite. Severely degenerate channels (n < 3, constant)
  produce zeros for the stats that need them rather than NaN.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Ordered feature names. Keep this list and the assembly order in sync;
# changing either invalidates existing memory banks.
FEATURE_NAMES: tuple[str, ...] = (
    "log_length",          # log10(median channel length + 1)
    "n_channels_log",      # log2(n_channels + 1)
    "missing_rate",        # fraction of NaN/inf across all channels
    "irregular_ts",        # 1.0 if timestamps present and non-uniformly spaced, else 0
    "mean_z",              # 0 (level removed by per-channel z-score before aggregating others)
    "std_log",             # log10(median per-channel std + eps)
    "iqr_over_std",        # robust spread / dispersion ratio (median across channels)
    "skewness",            # mean per-channel skewness
    "kurtosis",            # mean per-channel excess kurtosis
    "trend_slope_z",       # mean z-scored linear-fit slope
    "trend_r2",            # mean R^2 of linear fit (captures monotonicity)
    "acf_lag1",            # mean ACF at lag 1
    "acf_lag_sqrtn",       # mean ACF at lag floor(sqrt(N))
    "acf_lag_n_over_4",    # mean ACF at lag N//4
    "fft_top_freq_norm",   # dominant frequency (cycles/sample) of mean-detrended channel
    "fft_top_power_frac",  # share of non-DC spectral power held by top frequency
    "spectral_entropy",    # Shannon entropy of normalized spectrum, base-2 bits
    "changepoint_rate",    # cumulative-sum-based change rate per sample
    "mean_pairwise_corr",  # mean |corr| across distinct channel pairs (0 if univariate)
    "outlier_rate",        # fraction of points beyond 3 robust MADs
)
FINGERPRINT_DIM: int = len(FEATURE_NAMES)


_EPS = 1e-12


def _safe_arr(values: Any) -> np.ndarray:
    """Coerce a channel to a finite 1-D float array (NaN/inf removed)."""
    arr = np.asarray(values, dtype=float).ravel()
    mask = np.isfinite(arr)
    return arr[mask]


def _channel_features(arr: np.ndarray) -> dict[str, float]:
    """Single-channel descriptive features. Inputs assumed already finite."""
    n = arr.size
    if n < 3:
        return {
            "std": 0.0,
            "iqr_over_std": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "slope_z": 0.0,
            "r2": 0.0,
            "acf1": 0.0,
            "acf_sqrt": 0.0,
            "acf_q": 0.0,
            "fft_freq": 0.0,
            "fft_pfrac": 0.0,
            "spec_entropy": 0.0,
            "cp_rate": 0.0,
            "outlier": 0.0,
        }
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd < _EPS:
        # constant channel — degenerate, all shape features = 0
        return {
            "std": 0.0,
            "iqr_over_std": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "slope_z": 0.0,
            "r2": 0.0,
            "acf1": 0.0,
            "acf_sqrt": 0.0,
            "acf_q": 0.0,
            "fft_freq": 0.0,
            "fft_pfrac": 0.0,
            "spec_entropy": 0.0,
            "cp_rate": 0.0,
            "outlier": 0.0,
        }
    q25, q75 = float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))
    iqr = q75 - q25
    z = (arr - mu) / sd
    # Higher moments on the z-scored signal so they are scale-free.
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean()) - 3.0  # excess kurtosis

    # Linear trend on a unit-rescaled x-axis so the slope is comparable
    # across different-length series. r2 captures how monotonic it is.
    x = np.linspace(0.0, 1.0, n)
    slope, intercept = np.polyfit(x, arr, 1)
    fit = slope * x + intercept
    ss_res = float(((arr - fit) ** 2).sum())
    ss_tot = float(((arr - mu) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0
    slope_z = float(slope) / (sd + _EPS)

    # ACF at three diagnostically useful lags. Computed manually rather than
    # via statsmodels to keep deps minimal and behavior predictable.
    arr_c = arr - mu
    denom = float((arr_c * arr_c).sum()) + _EPS
    lag_sqrt = max(1, int(np.floor(np.sqrt(n))))
    lag_q = max(1, n // 4)

    def _acf_at(lag: int) -> float:
        if lag <= 0 or lag >= n:
            return 0.0
        return float((arr_c[:-lag] * arr_c[lag:]).sum() / denom)

    acf1 = _acf_at(1)
    acf_sqrt = _acf_at(lag_sqrt) if lag_sqrt != 1 else acf1
    acf_q = _acf_at(lag_q) if lag_q not in (1, lag_sqrt) else acf_sqrt

    # FFT on the detrended (mean-removed) signal. fft_freq is in cycles/sample
    # and lives in (0, 0.5]; fft_pfrac is in [0, 1] — high when periodic.
    if n >= 8:
        spec = np.abs(np.fft.rfft(arr_c)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0)
        if spec.size > 1:
            non_dc = spec[1:]
            total = float(non_dc.sum()) + _EPS
            idx = int(np.argmax(non_dc)) + 1
            fft_freq = float(freqs[idx])
            fft_pfrac = float(spec[idx] / total)
            # Spectral entropy of the normalized non-DC distribution.
            p = non_dc / total
            p = np.clip(p, _EPS, 1.0)
            spec_entropy = float(-(p * np.log2(p)).sum())
            # Normalize to [0, 1] by dividing by log2(K) so the scale is
            # comparable across different-length series.
            spec_entropy = spec_entropy / max(1.0, float(np.log2(p.size)))
        else:
            fft_freq, fft_pfrac, spec_entropy = 0.0, 0.0, 0.0
    else:
        fft_freq, fft_pfrac, spec_entropy = 0.0, 0.0, 0.0

    # Cumulative-sum changepoint heuristic: count zero-crossings of the
    # standardized cumulative sum of first differences, then normalize by n.
    diffs = np.diff(arr)
    if diffs.size > 1:
        cumdiff = np.cumsum(diffs - diffs.mean())
        sign = np.sign(cumdiff)
        zc = int((np.diff(sign) != 0).sum())
        cp_rate = float(zc) / float(n)
    else:
        cp_rate = 0.0

    # Robust outlier rate via MAD; 3 MAD ~ 2 sigma for gaussian, ok proxy.
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + _EPS
    outlier_rate = float((np.abs(arr - med) > 3.0 * mad).mean())

    return {
        "std": sd,
        "iqr_over_std": float(iqr) / (sd + _EPS),
        "skew": skew,
        "kurt": kurt,
        "slope_z": slope_z,
        "r2": r2,
        "acf1": acf1,
        "acf_sqrt": acf_sqrt,
        "acf_q": acf_q,
        "fft_freq": fft_freq,
        "fft_pfrac": fft_pfrac,
        "spec_entropy": spec_entropy,
        "cp_rate": cp_rate,
        "outlier": outlier_rate,
    }


def _mean_pairwise_abs_corr(channels: list[np.ndarray]) -> float:
    """Mean absolute Pearson correlation across distinct channel pairs.

    Channels of unequal length are truncated to the shortest. 0 for univariate
    or when fewer than 2 channels have nondegenerate variance.
    """
    if len(channels) < 2:
        return 0.0
    min_len = min(c.size for c in channels)
    if min_len < 3:
        return 0.0
    mat = np.vstack([c[:min_len] for c in channels])
    stds = mat.std(axis=1, ddof=1)
    keep = stds > _EPS
    if keep.sum() < 2:
        return 0.0
    mat = mat[keep]
    corr = np.corrcoef(mat)
    if corr.ndim != 2 or corr.shape[0] < 2:
        return 0.0
    # Upper triangle, off-diagonal
    iu = np.triu_indices(corr.shape[0], k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.abs(vals).mean())


def _irregular_timestamps(timestamps: list[str] | None) -> float:
    """Return 1.0 if timestamps are present and non-uniformly spaced, else 0.

    We try to parse them as datetimes; if parsing fails we treat them as
    regular (0). Considered irregular if the std of inter-step deltas exceeds
    1% of the mean delta.
    """
    if not timestamps or len(timestamps) < 3:
        return 0.0
    try:
        ts = np.asarray(timestamps, dtype="datetime64[ns]")
    except (TypeError, ValueError):
        return 0.0
    deltas = np.diff(ts).astype("int64").astype(float)
    if deltas.size == 0:
        return 0.0
    mu = deltas.mean()
    if mu <= 0:
        return 0.0
    return 1.0 if (deltas.std() / mu) > 0.01 else 0.0


def compute_fingerprint(series: dict[str, Any]) -> np.ndarray:
    """Compute the fixed-dim retrieval fingerprint for a series payload.

    Parameters
    ----------
    series
        Dict in the ``load_data`` shape: ``{"channels": {name: list[float]},
        "timestamps": list[str] | None, "meta": dict}``. Extra keys are
        ignored. Missing/empty channels result in a zero fingerprint.

    Returns
    -------
    np.ndarray of shape (FINGERPRINT_DIM,) and dtype float32. All entries are
    finite; degenerate inputs produce zeros for affected positions.
    """
    channels = series.get("channels") or {}
    timestamps = series.get("timestamps")

    chan_arrays: list[np.ndarray] = []
    total_pts = 0
    finite_pts = 0
    for vals in channels.values():
        raw = np.asarray(vals, dtype=float).ravel()
        total_pts += int(raw.size)
        arr = raw[np.isfinite(raw)]
        finite_pts += int(arr.size)
        if arr.size > 0:
            chan_arrays.append(arr)

    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

    if not chan_arrays:
        return fp

    # Structural features (always defined when at least one channel exists).
    lengths = np.array([c.size for c in chan_arrays], dtype=float)
    fp[FEATURE_NAMES.index("log_length")] = float(np.log10(np.median(lengths) + 1.0))
    fp[FEATURE_NAMES.index("n_channels_log")] = float(np.log2(len(chan_arrays) + 1.0))
    fp[FEATURE_NAMES.index("missing_rate")] = (
        0.0 if total_pts == 0 else 1.0 - finite_pts / total_pts
    )
    fp[FEATURE_NAMES.index("irregular_ts")] = _irregular_timestamps(timestamps)
    # mean_z is left at 0 by construction (channels are z-scored internally
    # for higher-moment features, so absolute level is not a retrieval signal).

    # Per-channel feature collection, then aggregate (mean) across channels.
    per_chan = [_channel_features(arr) for arr in chan_arrays]
    keys = (
        "std", "iqr_over_std", "skew", "kurt", "slope_z", "r2",
        "acf1", "acf_sqrt", "acf_q",
        "fft_freq", "fft_pfrac", "spec_entropy",
        "cp_rate", "outlier",
    )
    agg = {k: float(np.mean([pc[k] for pc in per_chan])) for k in keys}

    # std is log-scaled to avoid one outlier channel dominating the metric.
    fp[FEATURE_NAMES.index("std_log")] = float(np.log10(agg["std"] + 1.0))
    fp[FEATURE_NAMES.index("iqr_over_std")] = agg["iqr_over_std"]
    fp[FEATURE_NAMES.index("skewness")] = agg["skew"]
    fp[FEATURE_NAMES.index("kurtosis")] = agg["kurt"]
    fp[FEATURE_NAMES.index("trend_slope_z")] = agg["slope_z"]
    fp[FEATURE_NAMES.index("trend_r2")] = agg["r2"]
    fp[FEATURE_NAMES.index("acf_lag1")] = agg["acf1"]
    fp[FEATURE_NAMES.index("acf_lag_sqrtn")] = agg["acf_sqrt"]
    fp[FEATURE_NAMES.index("acf_lag_n_over_4")] = agg["acf_q"]
    fp[FEATURE_NAMES.index("fft_top_freq_norm")] = agg["fft_freq"]
    fp[FEATURE_NAMES.index("fft_top_power_frac")] = agg["fft_pfrac"]
    fp[FEATURE_NAMES.index("spectral_entropy")] = agg["spec_entropy"]
    fp[FEATURE_NAMES.index("changepoint_rate")] = agg["cp_rate"]
    fp[FEATURE_NAMES.index("outlier_rate")] = agg["outlier"]

    fp[FEATURE_NAMES.index("mean_pairwise_corr")] = _mean_pairwise_abs_corr(chan_arrays)

    # Replace any residual non-finite with 0 (defensive — shouldn't trigger).
    fp = np.where(np.isfinite(fp), fp, 0.0).astype(np.float32)
    return fp
