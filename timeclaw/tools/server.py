"""Factory for the in-memory FastMCP server bundled with each TimeClaw slot.

Each slot owns its own ``_SeriesState`` (closure-captured by the registered
tools), so when the eval harness preloads task data before invoking the
agent, that data is visible only to the same slot's analysis tools — even
when other slots are concurrently running other tasks.

``make_timeclaw_mcp_server`` returns ``(server, load_data_fn)``. The
``load_data_fn`` is a plain Python callable that mutates the slot's
``_SeriesState`` directly; it is intentionally NOT registered as an
``@mcp.tool`` so the agent never sees it in ``list_tools()`` and cannot
call it via tool_calls. Earlier versions exposed ``load_data`` as a tool
and observed the agent occasionally re-invoking it with malformed args
(missing ``channels``), corrupting state and erroring out the run. The
harness preload path now bypasses MCP entirely via the returned callable.

Agent-visible tool registry:

- ``list_channels()``: names of currently-loaded channels.
- ``series_overview()``: per-channel (n, min, max, mean) + timestamp count + meta.
- ``channel_stats(name)``: extended stats on one channel (n, min, max, mean,
  std, median, q25, q75).
- ``channel_values(name, start, end, stride)``: slice of raw values, capped
  at 500 per call to keep tool output bounded.
- ``compute_acf(name, max_lag)``: sample autocorrelation lag 0 to max_lag.
- ``detect_periodicity(name)``: FFT-based dominant period (in samples) + the
  fraction of non-DC spectral power it carries.
- ``find_peaks(name, prominence?)``: scipy-detected local maxima indices.
- ``arima_forecast(name, periods, p, d, q)``: ARIMA(p,d,q) point forecast for the next ``periods`` steps.
- ``portfolio_var(channels, weights, horizon, alpha, method)``: VaR of a weighted-price portfolio over ``horizon`` periods.
- ``portfolio_sharpe(channels, weights, risk_free, period_per_year)``: annualized Sharpe ratio of a weighted-price portfolio.
- ``capm_regression(asset_channel, market_channel)``: OLS regression of asset log returns on market log returns → ``{alpha, beta, r_squared}``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from fastmcp import FastMCP
from scipy.signal import find_peaks as _scipy_find_peaks


@dataclass
class _SeriesState:
    """Mutable per-slot store for the currently-loaded task's series data.

    Tools close over a single ``_SeriesState`` instance for the lifetime of
    one MCP server (one slot). ``load_data`` replaces the contents wholesale
    so consecutive tasks on the same slot stay isolated.
    """

    channels: dict[str, list[float]] = field(default_factory=dict)
    timestamps: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


HarnessLoadData = Callable[[dict[str, list[float]], list[str] | None, dict[str, Any] | None], None]


def make_timeclaw_mcp_server() -> tuple[FastMCP, HarnessLoadData]:
    """Build a FastMCP server with TimeClaw's analysis tool registry.

    Returns ``(server, load_data_fn)``. ``load_data_fn`` is a harness-only
    setter that mutates the server's private ``_SeriesState`` directly; it
    is NOT registered as an MCP tool so the agent cannot see or call it.
    """
    state = _SeriesState()
    mcp = FastMCP("timeclaw-tools")

    def load_data(
        channels: dict[str, list[float]],
        timestamps: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Replace the currently-loaded series. Harness-only, not an MCP tool.

        Called by the eval harness at the start of each sample via the slot's
        Python reference — bypasses the MCP transport entirely so a confused
        agent cannot accidentally re-invoke it with bad args.
        """
        state.channels = {k: [float(v) for v in vals] for k, vals in channels.items()}
        state.timestamps = [str(t) for t in timestamps] if timestamps else None
        state.meta = dict(meta) if meta else {}

    @mcp.tool
    def list_channels() -> list[str]:
        """Names of currently-loaded channels. Call this first to discover what
        data is available before drilling into individual channels."""
        return list(state.channels.keys())

    @mcp.tool
    def series_overview() -> dict[str, Any]:
        """Compact per-channel summary (n, min, max, mean) plus timestamp
        count and meta. Convenient orientation call before per-channel
        inspection."""
        summary: dict[str, Any] = {"channels": {}}
        for name, vals in state.channels.items():
            if not vals:
                summary["channels"][name] = {"n": 0, "min": None, "max": None, "mean": None}
                continue
            n = len(vals)
            summary["channels"][name] = {
                "n": n,
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / n,
            }
        summary["timestamps_n"] = len(state.timestamps) if state.timestamps else None
        summary["meta"] = dict(state.meta)
        return summary

    @mcp.tool
    def channel_stats(name: str) -> dict[str, Any]:
        """Extended descriptive statistics on a single channel:
        n, min, max, mean, std (sample), median, q25, q75."""
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded; call list_channels() first"}
        arr = np.asarray(ch, dtype=float)
        n = int(len(arr))
        if n == 0:
            return {"error": f"channel {name!r} is empty"}
        return {
            "n": n,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        }

    @mcp.tool
    def channel_values(
        name: str,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
    ) -> dict[str, Any]:
        """Slice raw values from a channel: ``channel[start:end:stride]``.

        Returns at most 500 elements per call (errors if the slice would be
        larger — increase ``stride`` or narrow the range). Use this to inspect
        specific segments; for whole-channel statistics prefer ``channel_stats``.
        """
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded"}
        if stride < 1:
            return {"error": "stride must be >= 1"}
        if end is None:
            end = len(ch)
        sl = ch[start:end:stride]
        if len(sl) > 500:
            return {
                "error": (
                    f"slice has {len(sl)} elements (max 500); "
                    f"increase stride or narrow [start, end)"
                )
            }
        return {"values": [float(v) for v in sl], "start": int(start), "end": int(end), "stride": int(stride)}

    @mcp.tool
    def compute_acf(name: str, max_lag: int = 20) -> dict[str, Any]:
        """Sample autocorrelation of one channel from lag 0 to ``max_lag``
        (inclusive). Useful for finding repeating structure / seasonality."""
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded"}
        arr = np.asarray(ch, dtype=float)
        n = len(arr)
        if n < 3:
            return {"error": "channel too short for ACF (need >= 3 points)"}
        max_lag = min(max(0, int(max_lag)), n - 1)
        arr_c = arr - arr.mean()
        denom = float((arr_c * arr_c).sum())
        if denom == 0.0:
            return {"acf": [1.0] + [0.0] * max_lag, "max_lag": max_lag}
        acf = [1.0]
        for lag in range(1, max_lag + 1):
            num = float((arr_c[:-lag] * arr_c[lag:]).sum())
            acf.append(num / denom)
        return {"acf": acf, "max_lag": max_lag}

    @mcp.tool
    def detect_periodicity(name: str) -> dict[str, Any]:
        """FFT-based dominant period in samples and the fraction of the non-DC
        spectral power it accounts for. Returns ``power_fraction`` in [0, 1] —
        values near 1 mean strongly periodic, values near 0 mean no clear period."""
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded"}
        arr = np.asarray(ch, dtype=float)
        n = len(arr)
        if n < 8:
            return {"error": "channel too short for FFT periodicity (need >= 8)"}
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0)
        if len(spec) <= 1:
            return {"error": "not enough frequency bins"}
        idx = int(np.argmax(spec[1:])) + 1
        f = float(freqs[idx])
        period = float(1.0 / f) if f > 0 else None
        total = float(spec[1:].sum())
        return {
            "dominant_period_samples": period,
            "power_fraction": float(spec[idx] / total) if total > 0 else 0.0,
        }

    @mcp.tool
    def find_peaks(
        name: str,
        prominence: float | None = None,
    ) -> dict[str, Any]:
        """Detect local maxima in a channel via scipy.signal.find_peaks.
        Returns peak indices, peak values, and total count. Pass an explicit
        ``prominence`` (in the channel's units) to suppress small wiggles."""
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded"}
        arr = np.asarray(ch, dtype=float)
        kwargs: dict[str, Any] = {}
        if prominence is not None:
            kwargs["prominence"] = float(prominence)
        idxs, _ = _scipy_find_peaks(arr, **kwargs)
        return {
            "indices": [int(i) for i in idxs],
            "values": [float(arr[i]) for i in idxs],
            "n_peaks": int(len(idxs)),
        }

    # ------------------------------------------------------------------
    # Forecasting + quantitative-finance tools
    # ------------------------------------------------------------------

    def _get_prices(name: str) -> np.ndarray | dict:
        ch = state.channels.get(name)
        if ch is None:
            return {"error": f"channel {name!r} not loaded; call list_channels() first"}
        arr = np.asarray(ch, dtype=float)
        if arr.size < 2:
            return {"error": f"channel {name!r} has <2 points; cannot compute returns"}
        return arr

    def _log_returns(prices: np.ndarray) -> np.ndarray:
        # log returns ignore zero/negative price by NaN propagation
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(prices))
        return r[np.isfinite(r)]

    @mcp.tool
    def arima_forecast(
        name: str,
        periods: int,
        p: int = 1,
        d: int = 0,
        q: int = 0,
    ) -> dict[str, Any]:
        """Fit an ARIMA(p,d,q) model to a single channel and return a point
        forecast for the next ``periods`` steps.

        Returns ``{"forecast": [...], "in_sample_aic": float, "order": [p,d,q]}``
        or ``{"error": ...}`` on fit failure. Defaults to ARIMA(1,0,0) — set
        ``d=1`` for non-stationary level series (e.g. raw prices), ``q`` > 0
        for moving-average structure. The agent is expected to choose
        ``order`` based on what it has observed (e.g. compute_acf result).
        """
        arr = _get_prices(name)
        if isinstance(arr, dict):
            return arr
        if periods < 1:
            return {"error": "periods must be >= 1"}
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return {"error": "statsmodels not available in this environment"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(arr, order=(int(p), int(d), int(q)))
                fit = model.fit()
                fc = fit.forecast(steps=int(periods))
            return {
                "forecast": [float(v) for v in np.asarray(fc).ravel()],
                "in_sample_aic": float(fit.aic),
                "order": [int(p), int(d), int(q)],
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ARIMA fit failed: {type(exc).__name__}: {exc}"}

    @mcp.tool
    def portfolio_var(
        channels: list[str],
        weights: list[float],
        horizon: int,
        alpha: float = 0.05,
        method: str = "historical",
    ) -> dict[str, Any]:
        """Compute the Value-at-Risk of a portfolio of price channels.

        Builds a weighted-price series from the listed channels (one per
        constituent, same length), converts to log returns, then estimates
        VaR at confidence level ``1 - alpha`` over a horizon of ``horizon``
        periods. Two methods:

          - ``"historical"``: empirical alpha-quantile of horizon-aggregated
            returns (recommended when sample size allows).
          - ``"parametric"``: normal-approximation, ``VaR = -(mu*h + z * sigma
            * sqrt(h))`` using per-period mean/std and z = Phi^{-1}(alpha).

        VaR is reported as a positive number representing the worst-case loss
        as a fraction of portfolio value. Smaller is safer. Returns
        ``{"var": float, "method": str, "horizon": int, "alpha": float,
        "n_returns": int}`` or ``{"error": ...}``.
        """
        if len(channels) != len(weights):
            return {"error": f"channels (n={len(channels)}) and weights (n={len(weights)}) length mismatch"}
        if not channels:
            return {"error": "channels list must be non-empty"}
        if horizon < 1:
            return {"error": "horizon must be >= 1"}
        if not (0.0 < alpha < 0.5):
            return {"error": f"alpha must be in (0, 0.5), got {alpha}"}
        price_mat = []
        for ch in channels:
            arr = _get_prices(ch)
            if isinstance(arr, dict):
                return arr
            price_mat.append(arr)
        # All channels must align in length
        n_min = min(len(a) for a in price_mat)
        prices = np.stack([a[:n_min] for a in price_mat], axis=1)  # (T, K)
        w = np.asarray(weights, dtype=float)
        # Weighted portfolio price = sum(w_i * price_i)
        port_price = prices @ w
        rets = _log_returns(port_price)
        if rets.size < horizon + 1:
            return {"error": f"only {rets.size} return points; need >= {horizon + 1} for horizon={horizon}"}
        if method == "historical":
            # Rolling horizon-sum returns, then alpha-quantile
            hsum = np.convolve(rets, np.ones(horizon), mode="valid")
            var_loss = float(-np.quantile(hsum, alpha))
        elif method == "parametric":
            from scipy.stats import norm
            mu = float(rets.mean())
            sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
            z = float(norm.ppf(alpha))
            var_loss = float(-(mu * horizon + z * sd * np.sqrt(horizon)))
        else:
            return {"error": f"unknown method {method!r}; use 'historical' or 'parametric'"}
        return {
            "var": var_loss,
            "method": method,
            "horizon": int(horizon),
            "alpha": float(alpha),
            "n_returns": int(rets.size),
        }

    @mcp.tool
    def portfolio_sharpe(
        channels: list[str],
        weights: list[float],
        risk_free: str | float = 0.0,
        period_per_year: int = 252,
    ) -> dict[str, Any]:
        """Annualized Sharpe ratio of a weighted portfolio.

        Combines listed price channels with the given weights, takes log
        returns, subtracts the per-period risk-free rate, then computes
        ``mean(excess) / std(excess) * sqrt(period_per_year)``.

        ``risk_free`` can be a constant per-period rate (float, e.g. 0.0
        when nominal) or the name of a channel containing per-period
        risk-free rates (same length as returns; will be aligned by
        truncation to the shorter of the two). ``period_per_year``
        annualizes — use 252 for daily, 8 for hourly intraday US session,
        etc. Returns ``{"sharpe": float, "mean_excess": float,
        "vol_excess": float, "period_per_year": int, "n": int}``.
        """
        if len(channels) != len(weights):
            return {"error": f"channels (n={len(channels)}) and weights (n={len(weights)}) length mismatch"}
        if not channels:
            return {"error": "channels list must be non-empty"}
        price_mat = []
        for ch in channels:
            arr = _get_prices(ch)
            if isinstance(arr, dict):
                return arr
            price_mat.append(arr)
        n_min = min(len(a) for a in price_mat)
        prices = np.stack([a[:n_min] for a in price_mat], axis=1)
        w = np.asarray(weights, dtype=float)
        rets = _log_returns(prices @ w)
        if rets.size < 2:
            return {"error": "fewer than 2 returns; cannot estimate Sharpe"}
        if isinstance(risk_free, str):
            rf_ch = state.channels.get(risk_free)
            if rf_ch is None:
                return {"error": f"risk_free channel {risk_free!r} not loaded"}
            rf = np.asarray(rf_ch, dtype=float)
            n = min(len(rets), len(rf))
            excess = rets[:n] - rf[:n]
        else:
            excess = rets - float(risk_free)
        mu = float(excess.mean())
        sd = float(excess.std(ddof=1))
        if sd == 0.0:
            return {"error": "zero variance in excess returns; Sharpe undefined"}
        sharpe = mu / sd * np.sqrt(float(period_per_year))
        return {
            "sharpe": float(sharpe),
            "mean_excess": mu,
            "vol_excess": sd,
            "period_per_year": int(period_per_year),
            "n": int(excess.size),
        }

    @mcp.tool
    def capm_regression(
        asset_channel: str,
        market_channel: str,
    ) -> dict[str, Any]:
        """OLS regression of asset log returns on market log returns (CAPM).

        Returns ``{"alpha": float, "beta": float, "r_squared": float, "n": int}``.
        Interpretation:
          - ``beta > 1``: asset more volatile than the market;
          - ``beta < 1``: asset less volatile;
          - ``alpha > 0``: asset outperformed the market after adjusting for
            beta exposure (positive risk-adjusted return);
          - ``alpha < 0``: underperformed.

        Series are aligned by truncation to the shorter common length.
        """
        a = _get_prices(asset_channel)
        if isinstance(a, dict):
            return a
        m = _get_prices(market_channel)
        if isinstance(m, dict):
            return m
        n_min = min(len(a), len(m))
        ar = _log_returns(a[:n_min])
        mr = _log_returns(m[:n_min])
        n = min(len(ar), len(mr))
        if n < 3:
            return {"error": f"only {n} aligned returns; need >= 3 for regression"}
        ar, mr = ar[:n], mr[:n]
        # OLS via numpy
        x = np.vstack([mr, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(x, ar, rcond=None)
        beta, alpha = float(coef[0]), float(coef[1])
        # R^2
        pred = x @ coef
        ss_res = float(((ar - pred) ** 2).sum())
        ss_tot = float(((ar - ar.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "alpha": alpha,
            "beta": beta,
            "r_squared": float(r2),
            "n": int(n),
        }

    return mcp, load_data
