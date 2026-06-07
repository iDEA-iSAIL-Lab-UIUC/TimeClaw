"""CiK (Context-is-Key) evaluator.

Tasks are grouped by domain (Table 2 in the CiK paper). For each task we ask
the agent for a probabilistic or point forecast over a fixed horizon and score
with RCRPS (using ``region_of_interest`` and ``metric_scaling`` from the task
record) plus point-forecast sanity metrics (sMAPE / MAAPE / MSE).
"""

from __future__ import annotations

import json
import math
import os.path as osp
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from timeclaw.agents import TimeClaw
from timeclaw.evaluation.common import (
    AgentCallResult,
    EvalWriter,
    RunConfig,
    call_agent_once,
    make_run_dir,
    run_tasks_concurrent,
    select_split,
    subsample_groups,
)
from timeclaw.memory import (
    MemoryBank,
    compute_fingerprint,
    embed_text,
    summarize_trajectory,
)
from timeclaw.evaluation.metrics import (
    aggregate_numeric,
    maape,
    mse,
    rcrps,
    smape,
)
from timeclaw.evaluation.parsers import parse_forecast
from timeclaw.evaluation.prompts import build_cik_prompt
from timeclaw.utils.path import dataset_sources_dir


# ---------------------------------------------------------------------------
# Domain mapping (verified against CiK paper Table 2; 71/71 task families)
# ---------------------------------------------------------------------------

CIK_TASK_TO_DOMAIN: dict[str, str] = {}

_RETAIL = [
    "ATMBuildingClosedTask",
    "ATMUnderPeriodicMaintenanceTaskWithConclusion",
    "ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit",
    "ATMUnderPeriodicMaintenanceTaskWithoutConclusion",
    "CashDepletedinATMScenarioTask",
    "IncreasedWithdrawalScenario",
]
_ENERGY = [
    "ElectricityIncreaseInPredictionTask",
    "ElectricityIncreaseInPredictionWithDistractorText",
    "ElectricityIncreaseInPredictionWithDistractorWithDates",
    "ElectricityIncreaseInPredictionWithSplitContext",
    "LongNewsElectricityIncreaseInPredictionTask",
    "MediumNewsElectricityIncreaseInPredictionTask",
    "ShortNewsElectricityIncreaseInPredictionTask",
]
_TRAFFIC = [
    "BoundedPredConstraintsBasedOnPredQuantilesTask",
    "DecreaseInTrafficInPredictionTask",
    "ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow",
    "OraclePredUnivariateConstraintsTask",
    "SensorMaintenanceInPredictionTask",
    "SensorPeriodicMaintenanceTask",
    "SensorSpikeTask",
    "SensorTrendAccumulationTask",
]
_MECHANICS = [
    "ExplicitPressureFromSpeedTask",
    "ImplicitPressureFromSpeedTask",
    "SpeedFromLoadTask",
]
_ECONOMICS = [
    "UnemploymentCountyUsingExplicitMultipleStateData",
    "UnemploymentCountyUsingMultipleStateData",
    "UnemploymentCountyUsingSingleStateData",
]
_SYNTHETIC = [
    "FullCausalContextExplicitEquationBivarLinSVAR",
    "FullCausalContextImplicitEquationBivarLinSVAR",
    "MinimalCausalContextBivarLinSVAR",
]
_CLIMATOLOGY = [
    "DiffuseHorizontalIrradianceFromCloudStatus",
    "DirectNormalIrradianceFromClearsky",
    "DirectNormalIrradianceFromCloudStatus",
    "ExplicitDiffuseHorizontalIrradianceFromCloudStatus",
    "ExplicitDirectNormalIrradianceFromCloudStatus",
    "ExplicitSimilarLocationDaySolarForecastTask",
    "GlobalHorizontalIrradianceFromClearsky",
    "LocaleInfoHalfDaySolarForecastTask",
    "MinimalInfoHalfDaySolarForecastTask",
    "SimilarLocationDaySolarForecastTask",
    "SimilarLocationWithReferenceDaySolarForecastTask",
    "ZenithInfoHalfDaySolarForecastTask",
]

for name in _RETAIL: CIK_TASK_TO_DOMAIN[name] = "Retail"
for name in _ENERGY: CIK_TASK_TO_DOMAIN[name] = "Energy"
for name in _TRAFFIC: CIK_TASK_TO_DOMAIN[name] = "Traffic"
for name in _MECHANICS: CIK_TASK_TO_DOMAIN[name] = "Mechanics"
for name in _ECONOMICS: CIK_TASK_TO_DOMAIN[name] = "Economics"
for name in _SYNTHETIC: CIK_TASK_TO_DOMAIN[name] = "Synthetic"
for name in _CLIMATOLOGY: CIK_TASK_TO_DOMAIN[name] = "Climatology"


def _classify_montreal_fire(task_name: str) -> str | None:
    if task_name.startswith("MontrealFire"):
        return "Public Safety"
    return None


def cik_domain_of(task_name: str) -> str:
    if task_name in CIK_TASK_TO_DOMAIN:
        return CIK_TASK_TO_DOMAIN[task_name]
    d = _classify_montreal_fire(task_name)
    return d or "UNKNOWN"


# ---------------------------------------------------------------------------
# Context-type mapping (5 context types from CiK paper).
#
# Unlike domain (1-to-1: each task family belongs to exactly one domain),
# context is many-to-many: a single task family can carry History +
# Intemporal + Covariates + Causal at once (e.g. the MontrealFire causal-
# confounding tasks). So we maintain ``CIK_CONTEXTS[context] -> set(names)``
# and let ``cik_contexts_of(name)`` return every context the family belongs
# to. ``--cik-context a,b,c`` is a UNION filter at run time.
# ---------------------------------------------------------------------------

# Shared MontrealFire groupings (reused across context lists).
_MF_CAUSAL_CONFOUNDING = [
    "MontrealFireFieldAndTrashNeutralToneExplicitCausalConfoundingTask",
    "MontrealFireFieldAndGasNeutralToneExplicitCausalConfoundingTask",
    "MontrealFireTrashAndNauticalNeutralToneExplicitCausalConfoundingTask",
    "MontrealFireTrashAndBicycleNeutralToneExplicitCausalConfoundingTask",
    "MontrealFireFieldAndTrashConvincingToneExplicitCausalConfoundingTask",
    "MontrealFireFieldAndGasConvincingToneExplicitCausalConfoundingTask",
    "MontrealFireTrashAndNauticalConvincingToneExplicitCausalConfoundingTask",
    "MontrealFireTrashAndBicycleConvincingToneExplicitCausalConfoundingTask",
    "MontrealFireFieldAndTrashNeutralToneImplicitCausalConfoundingTask",
    "MontrealFireFieldAndGasNeutralToneImplicitCausalConfoundingTask",
    "MontrealFireTrashAndNauticalNeutralToneImplicitCausalConfoundingTask",
    "MontrealFireTrashAndBicycleNeutralToneImplicitCausalConfoundingTask",
    "MontrealFireFieldAndTrashConvincingToneImplicitCausalConfoundingTask",
    "MontrealFireFieldAndGasConvincingToneImplicitCausalConfoundingTask",
    "MontrealFireTrashAndNauticalConvincingToneImplicitCausalConfoundingTask",
    "MontrealFireTrashAndBicycleConvincingToneImplicitCausalConfoundingTask",
]
_MF_SHORT_HISTORY = [
    "MontrealFireFieldFireExplicitShortHistoryTask",
    "MontrealFireFieldFireImplicitShortHistoryTask",
    "MontrealFireTrashFireExplicitShortHistoryTask",
    "MontrealFireTrashFireImplicitShortHistoryTask",
    "MontrealFireNauticalRescueExplicitShortHistoryTask",
    "MontrealFireNauticalRescueImplicitShortHistoryTask",
    "MontrealFireIceRescueExplicitShortHistoryTask",
    "MontrealFireIceRescueImplicitShortHistoryTask",
]
_MF_ANALOGY = [
    "MontrealFireNauticalRescueAnalogyFullLocalizationMaybeWaterTask",
    "MontrealFireNauticalRescueAnalogyTargetLocalizationMaybeWaterTask",
]

_CTX_HISTORY = [
    "ZenithInfoHalfDaySolarForecastTask",
    *_MF_CAUSAL_CONFOUNDING,
    *_MF_SHORT_HISTORY,
]

_CTX_FUTURE = [
    "ElectricityIncreaseInPredictionTask",
    "ElectricityIncreaseInPredictionWithDistractorText",
    "ElectricityIncreaseInPredictionWithDistractorWithDates",
    "ElectricityIncreaseInPredictionWithSplitContext",
    "ShortNewsElectricityIncreaseInPredictionTask",
    "MediumNewsElectricityIncreaseInPredictionTask",
    "LongNewsElectricityIncreaseInPredictionTask",
    "CashDepletedinATMScenarioTask",
    "ATMBuildingClosedTask",
    "IncreasedWithdrawalScenario",
    "DecreaseInTrafficInPredictionTask",
    "OraclePredUnivariateConstraintsTask",
    "BoundedPredConstraintsBasedOnPredQuantilesTask",
    "SensorMaintenanceInPredictionTask",
    *_MF_CAUSAL_CONFOUNDING,
    "UnemploymentCountyUsingSingleStateData",
    "UnemploymentCountyUsingMultipleStateData",
    "UnemploymentCountyUsingExplicitMultipleStateData",
]

_CTX_INTEMPORAL = [
    "CashDepletedinATMScenarioTask",
    "ATMBuildingClosedTask",
    "ATMUnderPeriodicMaintenanceTaskWithConclusion",
    "ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit",
    "ATMUnderPeriodicMaintenanceTaskWithoutConclusion",
    "IncreasedWithdrawalScenario",
    "SpeedFromLoadTask",
    "ExplicitPressureFromSpeedTask",
    "ImplicitPressureFromSpeedTask",
    "MinimalInfoHalfDaySolarForecastTask",
    "LocaleInfoHalfDaySolarForecastTask",
    "ZenithInfoHalfDaySolarForecastTask",
    "SimilarLocationDaySolarForecastTask",
    "ExplicitSimilarLocationDaySolarForecastTask",
    "SimilarLocationWithReferenceDaySolarForecastTask",
    "ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "DirectNormalIrradianceFromCloudStatus",
    "ExplicitDirectNormalIrradianceFromCloudStatus",
    "DiffuseHorizontalIrradianceFromCloudStatus",
    "ExplicitDiffuseHorizontalIrradianceFromCloudStatus",
    "GlobalHorizontalIrradianceFromClearsky",
    "DirectNormalIrradianceFromClearsky",
    *_MF_ANALOGY,
    *_MF_CAUSAL_CONFOUNDING,
    *_MF_SHORT_HISTORY,
]

_CTX_COVARIATES = [
    "ElectricityIncreaseInPredictionTask",
    "ElectricityIncreaseInPredictionWithDistractorText",
    "ElectricityIncreaseInPredictionWithDistractorWithDates",
    "ElectricityIncreaseInPredictionWithSplitContext",
    "ShortNewsElectricityIncreaseInPredictionTask",
    "MediumNewsElectricityIncreaseInPredictionTask",
    "LongNewsElectricityIncreaseInPredictionTask",
    "ATMUnderPeriodicMaintenanceTaskWithConclusion",
    "ATMUnderPeriodicMaintenanceTaskWithConclusionLessExplicit",
    "ATMUnderPeriodicMaintenanceTaskWithoutConclusion",
    "IncreasedWithdrawalScenario",
    "DecreaseInTrafficInPredictionTask",
    "SensorMaintenanceInPredictionTask",
    "SensorPeriodicMaintenanceTask",
    "SensorTrendAccumulationTask",
    "SensorSpikeTask",
    "SpeedFromLoadTask",
    "ExplicitPressureFromSpeedTask",
    "ImplicitPressureFromSpeedTask",
    "MinimalCausalContextBivarLinSVAR",
    "FullCausalContextImplicitEquationBivarLinSVAR",
    "FullCausalContextExplicitEquationBivarLinSVAR",
    "ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "ExplicitWithDatesAndDaysTrafficForecastTaskwithHolidaysInPredictionWindow",
    "DirectNormalIrradianceFromCloudStatus",
    "ExplicitDirectNormalIrradianceFromCloudStatus",
    "DiffuseHorizontalIrradianceFromCloudStatus",
    "ExplicitDiffuseHorizontalIrradianceFromCloudStatus",
    "GlobalHorizontalIrradianceFromClearsky",
    "DirectNormalIrradianceFromClearsky",
    *_MF_ANALOGY,
    *_MF_CAUSAL_CONFOUNDING,
    "UnemploymentCountyUsingSingleStateData",
    "UnemploymentCountyUsingMultipleStateData",
    "UnemploymentCountyUsingExplicitMultipleStateData",
]

_CTX_CAUSAL = [
    "SpeedFromLoadTask",
    "ExplicitPressureFromSpeedTask",
    "ImplicitPressureFromSpeedTask",
    "MinimalCausalContextBivarLinSVAR",
    "FullCausalContextImplicitEquationBivarLinSVAR",
    "FullCausalContextExplicitEquationBivarLinSVAR",
    *_MF_CAUSAL_CONFOUNDING,
]

CIK_CONTEXTS: dict[str, set[str]] = {
    "history": set(_CTX_HISTORY),
    "future": set(_CTX_FUTURE),
    "intemporal": set(_CTX_INTEMPORAL),
    "covariates": set(_CTX_COVARIATES),
    "causal": set(_CTX_CAUSAL),
}
# Canonical iteration order — matters for stable summary output.
CIK_CONTEXT_NAMES: tuple[str, ...] = (
    "history", "future", "intemporal", "covariates", "causal",
)


def cik_contexts_of(task_name: str) -> list[str]:
    """Return every context type the task family belongs to (possibly empty)."""
    return [c for c in CIK_CONTEXT_NAMES if task_name in CIK_CONTEXTS[c]]


def _parse_cik_context_arg(arg: str) -> list[str]:
    """Validate ``--cik-context`` value. ``"all"`` (or empty) → no filter."""
    if not arg or arg.strip().lower() == "all":
        return []
    out: list[str] = []
    for tok in arg.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in CIK_CONTEXTS:
            raise ValueError(
                f"unknown CiK context {tok!r}; choose from "
                f"{list(CIK_CONTEXT_NAMES) + ['all']}"
            )
        if tok not in out:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cik_records() -> list[dict]:
    path = osp.join(dataset_sources_dir, "CiK", "all_tasks.json")
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    for r in records:
        r["_domain"] = cik_domain_of(r["name"])
        r["_contexts"] = cik_contexts_of(r["name"])
        r["_task_id"] = f"{r['name']}__seed{r.get('seed', 0)}"
    return records


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def _extract_future_ground_truth(record: dict) -> tuple[list[float], int]:
    """Flatten the future_time field into (values, horizon).

    CiK stores future_time as ``{channel -> {ts -> value}}``. We concatenate
    channel-by-channel to match the forecast vector emitted by the agent.
    """
    future_time = record.get("future_time")
    if isinstance(future_time, str):
        try:
            future_time = json.loads(future_time)
        except Exception:
            future_time = {}
    flat: list[float] = []
    horizon = 0
    if isinstance(future_time, dict):
        for col_values in future_time.values():
            if isinstance(col_values, dict):
                vals = list(col_values.values())
                flat.extend([float(v) for v in vals])
                horizon = len(vals)  # per-channel horizon (assumes same length)
    return flat, horizon


def _prepare_series(record: dict) -> dict[str, Any]:
    """Normalize a CiK record's ``past_time`` into the load_data shape.

    Returns ``{"channels": {col: list[float]}, "timestamps": list[str] | None,
    "meta": {}}``. Used both for fingerprint keying and as the harness
    preload payload for the agent's MCP server — once preloaded the agent
    can call ``channel_stats`` / ``compute_acf`` / ``detect_periodicity``
    on past_time during its analysis. The prompt continues to inline
    past_time as JSON too, so the agent has both surfaces available.
    """
    pt = record.get("past_time")
    if isinstance(pt, str):
        try:
            pt = json.loads(pt)
        except Exception:
            pt = None
    channels: dict[str, list[float]] = {}
    timestamps: list[str] | None = None
    if isinstance(pt, dict):
        for col, ts_to_val in pt.items():
            if not isinstance(ts_to_val, dict):
                continue
            if timestamps is None:
                timestamps = [str(k) for k in ts_to_val.keys()]
            vals: list[float] = []
            for k in (timestamps if timestamps is not None else ts_to_val.keys()):
                v = ts_to_val.get(k)
                if v is None:
                    vals.append(float("nan"))
                else:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        vals.append(float("nan"))
            channels[str(col)] = vals
    return {"channels": channels, "timestamps": timestamps, "meta": {}}


# Bounded penalty for parse-fail (so failures count in the mean rather
# than being silently dropped by aggregate_numeric's NaN skip). sMAPE and
# MAAPE have natural mathematical upper bounds; RCRPS is unbounded so we
# pick a value large enough to be clearly "fail" but small enough not to
# wreck the mean when one record fails — CiK paper's typical RCRPS lives
# in [0, ~3], so 5.0 is "absolutely worst case, still plausible".
PARSE_FAIL_SMAPE = 200.0          # sMAPE in [0, 200] by construction
PARSE_FAIL_MAAPE = math.pi / 2    # MAAPE in [0, pi/2] by construction
RCRPS_CAP = 5.0                   # also applied to valid records to bound outliers


def _score_one(record: dict, parsed: dict | None) -> dict[str, float]:
    y_true, horizon = _extract_future_ground_truth(record)
    roi = record.get("region_of_interest") or []
    metric_scaling = float(record.get("metric_scaling") or 1.0)

    if parsed is None or horizon == 0:
        return {
            "smape": PARSE_FAIL_SMAPE,
            "maape": PARSE_FAIL_MAAPE,
            "mse": float("nan"),    # MSE has no natural bound; leave as NaN
            "rcrps": RCRPS_CAP,
            "horizon": horizon,
            "forecast_kind": "none",
        }

    if parsed.get("kind") == "point":
        y_pred = parsed["values"]
        # Pad / truncate per-channel forecast to match y_true if needed.
        if len(y_pred) != len(y_true):
            # Best-effort: take first len(y_true) numbers
            y_pred = (y_pred + [y_pred[-1]] * max(0, len(y_true) - len(y_pred)))[: len(y_true)]
        rcrps_val = rcrps(y_true[:horizon], y_pred[:horizon], roi, metric_scaling)
        return {
            "smape": smape(y_true, y_pred),
            "maape": maape(y_true, y_pred),
            "mse": mse(y_true, y_pred),
            "rcrps": _clip_rcrps(rcrps_val),
            "horizon": horizon,
            "forecast_kind": "point",
        }

    # Quantile forecast — use median for point metrics, full RCRPS for proper score.
    levels = parsed.get("levels") or []
    samples = parsed.get("samples") or []
    # Median index (closest to 0.5)
    if levels:
        median_idx = int(np.argmin([abs(q - 0.5) for q in levels]))
        median_forecast = [row[median_idx] for row in samples]
    else:
        median_forecast = []
    if len(median_forecast) != len(y_true):
        median_forecast = (median_forecast + [0.0] * max(0, len(y_true) - len(median_forecast)))[: len(y_true)]

    rcrps_val = rcrps(y_true[:horizon], {"levels": levels, "samples": samples[:horizon]}, roi, metric_scaling)
    return {
        "smape": smape(y_true, median_forecast),
        "maape": maape(y_true, median_forecast),
        "mse": mse(y_true, median_forecast),
        "rcrps": _clip_rcrps(rcrps_val),
        "horizon": horizon,
        "forecast_kind": "quantile",
    }


def _clip_rcrps(v: float) -> float:
    """Bound a single record's RCRPS at RCRPS_CAP so one bad forecast can't
    wreck the mean. NaN passes through (will be NaN-skipped by aggregate)."""
    if v is None or math.isnan(v):
        return v
    return float(min(v, RCRPS_CAP))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _cik_take_first_per_family(
    records: list[dict], n: int
) -> list[dict]:
    """Take the first n records of each CiK family by ascending seed.

    CiK has 71 task families × 5 seeds = 355 records. ``--cik-samples N`` uses
    this to pick a deterministic, easily-reasonable subset (e.g. N=3 → seeds
    0/1/2 of every family) instead of the random shuffle ``subsample_groups``
    would otherwise apply.
    """
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fam[r["name"]].append(r)
    out: list[dict] = []
    for fam in sorted(by_fam):
        fam_recs = sorted(by_fam[fam], key=lambda r: int(r.get("seed", 0)))
        out.extend(fam_recs[:n])
    return out


def _cik_split_first_per_family(
    records: list[dict], train_n: int
) -> tuple[list[dict], list[dict]]:
    """Within each CiK family, the first ``train_n`` seeds go to train, rest test."""
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fam[r["name"]].append(r)
    train: list[dict] = []
    test: list[dict] = []
    for fam in sorted(by_fam):
        fam_recs = sorted(by_fam[fam], key=lambda r: int(r.get("seed", 0)))
        train.extend(fam_recs[:train_n])
        test.extend(fam_recs[train_n:])
    return train, test


def _cik_context_text(record: dict) -> str:
    """Join CiK's NL context fields for text-embedding similarity."""
    parts = []
    for f in ("background", "scenario", "constraints"):
        v = (record.get(f) or "").strip()
        if v:
            parts.append(v)
    return " ".join(parts) or "(empty)"


async def run_cik(
    agent: TimeClaw,
    *,
    mode: str = "full",
    train_ratio: float = 0.5,
    split_seed: int = 2026,
    k_neighbors: int = 3,
    retrieve_same_family_only: bool = False,
    text_filter_size: int = 0,
    cik_samples: int | None = None,
    cik_train_samples: int | None = None,
    cik_context: str = "all",
    num_workers: int = 4,
    ratio: float = 1.0,
    seed: int = 42,
    want_quantiles: bool = True,
) -> str:
    """Run the CiK evaluation and write results to disk. Returns the run dir."""
    records = load_cik_records()
    # Context-type filter (union across requested contexts). Applied BEFORE
    # subsampling so --ratio / --cik-samples behave consistently inside the
    # filtered set (e.g. "causal-only with 2 seeds per family" stays well
    # defined). "all" / empty → no filter.
    ctx_filter = _parse_cik_context_arg(cik_context)
    if ctx_filter:
        wanted: set[str] = set().union(*(CIK_CONTEXTS[c] for c in ctx_filter))
        before = len(records)
        records = [r for r in records if r["name"] in wanted]
        print(
            f"[cik] --cik-context={','.join(ctx_filter)}: "
            f"kept {len(records)}/{before} records "
            f"across {len({r['name'] for r in records})} families.",
            flush=True,
        )
    # Subsample within each (task_family, ) group — seeds inside one family count as one group.
    group_key_fn = lambda r: r["name"]
    if cik_samples is not None:
        # Deterministic CiK-specific override: first N seeds per family.
        sampled = _cik_take_first_per_family(records, cik_samples)
        print(
            f"[cik] --cik-samples={cik_samples} overrides --ratio "
            f"({ratio}); picked first {cik_samples} seed(s) per family -> "
            f"{len(sampled)} records.",
            flush=True,
        )
    else:
        sampled = subsample_groups(records, group_key_fn=group_key_fn, ratio=ratio, seed=seed)
    if cik_train_samples is not None:
        if cik_samples is None:
            raise ValueError("--cik-train-samples requires --cik-samples to be set")
        train, test = _cik_split_first_per_family(sampled, cik_train_samples)
        if mode == "full":
            selected = sampled
        else:
            selected = train if mode == "train" else test
        split_stats = {
            "n_subsampled": len(sampled),
            "n_train": len(train),
            "n_test": len(test),
        }
        print(
            f"[cik] --cik-train-samples={cik_train_samples} overrides --train-ratio "
            f"({train_ratio}); split -> train={len(train)} test={len(test)} "
            f"(mode={mode!r} selects {len(selected)} records).",
            flush=True,
        )
    else:
        selected, split_stats = select_split(
            sampled, group_key_fn, mode=mode, train_ratio=train_ratio, split_seed=split_seed,
        )

    # Encode the context filter (if any) into the run-dir name so two runs
    # that only differ by --cik-context don't land on the same folder. Join
    # multiple contexts with '-' to keep it path-safe (commas are OS-legal
    # but read ugly in `ls`).
    subset_label = "-".join(ctx_filter) if ctx_filter else None
    # CiK has 5 seeds per family; --cik-samples N is the deterministic-take-
    # first-N override of --ratio. Convert to an effective ratio for the
    # bank/dir naming so cik_samples=2 vs cik_samples=5 vs --ratio=1.0 land
    # in distinct bank dirs (otherwise their train splits could contaminate
    # each other's test splits at retrieval time).
    effective_ratio = (cik_samples / 5) if cik_samples is not None else ratio
    # k_neighbors only affects test mode (retrieval); tag the dir with it so
    # the k=0 baseline and k=3 memory run don't share a folder name.
    run_dir = make_run_dir(
        "cik", agent.model, mode,
        ratio=effective_ratio,
        subset=subset_label,
        k=k_neighbors if mode == "test" else None,
    )
    writer = EvalWriter(
        run_dir,
        RunConfig(
            benchmark="cik",
            model=agent.model,
            mode=mode,
            ratio=ratio,
            seed=seed,
            train_ratio=train_ratio if mode != "full" else 0.0,
            split_seed=split_seed,
            num_workers=num_workers,
            timestamp=osp.basename(run_dir),
            extra={
                "want_quantiles": want_quantiles,
                "n_records": len(selected),
                "n_total": len(records),
                "retrieve_same_family_only": retrieve_same_family_only,
                "cik_samples": cik_samples,
                "cik_train_samples": cik_train_samples,
                "cik_context": ctx_filter or "all",
                **split_stats,
            },
        ),
    )

    bank = None
    if mode in ("train", "test"):
        bank = MemoryBank.for_split(
            benchmark="cik",
            model=agent.model,
            split_seed=split_seed,
            train_ratio=train_ratio,
            ratio=effective_ratio,
        )
        if mode == "test" and bank.is_empty:
            print(
                f"[cik] mode=test: memory bank at {bank.bank_dir} is empty — "
                "retrieval will be a no-op. Run --mode train first to populate it.",
                flush=True,
            )

    async def _task(record: dict) -> dict:
        # GT for CiK is the future_time dict — the suffix helper JSON-dumps
        # it. The agent sees both the historical past_time and the target
        # future_time when training; its trajectory should demonstrate how
        # past structure leads to the target shape.
        gt_for_prompt = record.get("future_time") if mode == "train" else None
        series = _prepare_series(record)
        # Test mode: deterministic fingerprint -> top-k references injected
        # before the question.
        refs = None
        if mode == "test" and bank is not None and not bank.is_empty:
            fp = compute_fingerprint(series)
            text_query_emb = None
            if text_filter_size > 0:
                text_query_emb = embed_text(_cik_context_text(record))
            neighbors = bank.retrieve_topk(
                fp,
                k=k_neighbors,
                family_filter=record["name"] if retrieve_same_family_only else None,
                text_query_embedding=text_query_emb,
                text_filter_size=text_filter_size if text_filter_size > 0 else None,
            )
            refs = [summarize_trajectory(n.record) for n in neighbors]
        prompt = build_cik_prompt(
            record,
            want_quantiles=want_quantiles,
            ground_truth=gt_for_prompt,
            memory_references=refs,
            with_tools_hint=True,
        )
        # Determine expected_length = sum of channel-wise horizons (for parsing)
        y_true, horizon = _extract_future_ground_truth(record)
        # Preload past_time into the agent's MCP server so it can use the
        # inspection tools on the historical channels.
        result = await call_agent_once(agent, prompt, series=series)
        parsed = parse_forecast(result.text or "", expected_length=len(y_true))
        scored = _score_one(record, parsed)
        line = {
            "task_id": record["_task_id"],
            "name": record["name"],
            "domain": record["_domain"],
            "contexts": record["_contexts"],
            "seed": record.get("seed"),
            "metrics": scored,
            "prediction": parsed,
            "raw_text": result.text,
            "token_info": result.token_info,
            "elapsed_s": result.elapsed,
            "error": result.error,
        }
        await writer.write_prediction(line, trajectory=result.raw_response)
        if mode == "train" and bank is not None and result.error is None:
            log_record: dict[str, Any] = {
                "task_id": record["_task_id"],
                "family_key": record["name"],
                "fingerprint": compute_fingerprint(series),
                "prompt": prompt,
                "ground_truth": record.get("future_time"),
                "model_answer": result.text,
                "model_answer_parsed": parsed,
                "trajectory": result.raw_response,
                "model": agent.model,
            }
            # Embed the NL context once at log time so test-mode retrievals
            # can use the bank without re-embedding the training set every
            # time. Only spend the API call when text retrieval is enabled.
            if text_filter_size > 0:
                log_record["text_embedded"] = _cik_context_text(record)
                log_record["text_embedding"] = embed_text(
                    log_record["text_embedded"]
                ).tolist()
            await bank.log(log_record)
        return line

    results = await run_tasks_concurrent(selected, _task, num_workers=num_workers, desc="CiK")

    # Aggregate by domain (1-to-1) and by context (many-to-many: a record
    # contributes to every context its family belongs to, so per_context
    # bucket sizes may sum to > len(results)).
    summary: dict[str, Any] = {"per_domain": {}, "per_context": {}, "overall": {}}
    by_domain: dict[str, list[dict]] = defaultdict(list)
    by_context: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)
        for c in (r.get("contexts") or []):
            by_context[c].append(r)

    def _agg(items: list[dict]) -> dict[str, Any]:
        return {
            "n_tasks": len(items),
            "n_failed_to_parse": sum(1 for it in items if it["prediction"] is None),
            "n_api_errors": sum(1 for it in items if it["error"]),
            "smape": aggregate_numeric([it["metrics"]["smape"] for it in items]),
            "maape": aggregate_numeric([it["metrics"]["maape"] for it in items]),
            "mse_median": aggregate_numeric(
                [it["metrics"]["mse"] for it in items], how="median"
            ),
            "rcrps": aggregate_numeric([it["metrics"]["rcrps"] for it in items]),
            "mean_elapsed_s": aggregate_numeric([it["elapsed_s"] for it in items]),
            "total_tokens": sum(
                (it["token_info"] or {}).get("total_tokens", 0) for it in items
            ),
        }

    for dom, items in by_domain.items():
        summary["per_domain"][dom] = _agg(items)
    # Iterate in canonical context order so summary.json reads top→bottom as
    # history, future, intemporal, covariates, causal — even if some buckets
    # are empty in a filtered run, we still emit them so downstream parsing
    # has stable keys.
    for ctx in CIK_CONTEXT_NAMES:
        items = by_context.get(ctx, [])
        summary["per_context"][ctx] = _agg(items)
    summary["overall"] = {
        "n_tasks": len(results),
        "n_failed_to_parse": sum(1 for r in results if r["prediction"] is None),
        "n_api_errors": sum(1 for r in results if r["error"]),
        "smape": aggregate_numeric([r["metrics"]["smape"] for r in results]),
        "maape": aggregate_numeric([r["metrics"]["maape"] for r in results]),
        "mse_median": aggregate_numeric(
            [r["metrics"]["mse"] for r in results], how="median"
        ),
        "rcrps": aggregate_numeric([r["metrics"]["rcrps"] for r in results]),
        "mean_elapsed_s": aggregate_numeric([r["elapsed_s"] for r in results]),
        "total_tokens": sum(
            (r["token_info"] or {}).get("total_tokens", 0) for r in results
        ),
    }
    summary["mode"] = mode
    if bank is not None:
        summary["memory_bank"] = {
            "dir": str(bank.bank_dir),
            "n_records": len(bank),
            "k_neighbors": k_neighbors if mode == "test" else None,
        }
    writer.finalize_summary(summary)
    return run_dir
