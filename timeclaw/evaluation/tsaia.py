"""TSAIA evaluator.

TSAIA ships two CSV files: analysis_questions.csv (904 free-form tasks with
`executor_variables` and per-task `constraint` strings) and
multiple_choice.csv (150 MC questions). We group the 30 distinct
``question_type`` values into 5 categories at a granularity similar to
TSRBench's 4-category layout:

  - anomaly       : climate_anomaly, climate_anomaly_large, ecg_anomaly,
                    energy_anomaly
  - causal        : causal_knowledge, causal_relation
  - forecasting   : easy_stock-{price,trend,volatility},
                    electricity_prediction(_large|_single)-{ramp,variability,
                    max_load,min_load}
  - decision      : stock_investment, stock_ir_estimation, MarketAB-alpha,
                    MarketAB-beta, VaR-confidence_level
  - risk_metric   : stock_rv_estimation-{sharpe,sortino,calmar,maxdd,
                    annualized return,annualized volatility}, SharpeRatio (MC)

For each task we score:
  - MC items: exact-letter accuracy.
  - analysis items: parse the agent's final ``<answer>...</answer>`` payload
    and compare against ``ground_truth_data``. We compute MAE on a per-task
    basis when both sides are numeric, MAPE-style relative error when scales
    vary, and exact-match for short categorical answers; the per-family
    aggregation reports whichever is appropriate.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os.path as osp
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from timeclaw.agents import TimeClaw
from timeclaw.evaluation.common import (
    EvalWriter,
    RunConfig,
    call_agent_once,
    make_run_dir,
    run_tasks_concurrent,
    select_split,
    subsample_groups,
)
from timeclaw.evaluation.metrics import accuracy, aggregate_numeric, mae, smape
from timeclaw.evaluation.parsers import parse_mc_letter, parse_number
from timeclaw.evaluation.prompts import (
    build_tsaia_analysis_prompt,
    build_tsaia_mc_prompt_with_tools,
)
from timeclaw.memory import (
    MemoryBank,
    compute_fingerprint,
    embed_text,
    summarize_trajectory,
)
from timeclaw.utils.path import dataset_sources_dir

# csv.field_size_limit accepts a C long, which is 32-bit on Windows even on
# 64-bit Python. sys.maxsize (2**63-1) overflows there, so step down until the
# call is accepted.
_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max)
        break
    except OverflowError:
        _max = _max // 10


# ---------------------------------------------------------------------------
# Question-type → category mapping (re-grouped at TSRBench-like granularity)
# ---------------------------------------------------------------------------

def tsaia_category_of(question_type: str) -> str:
    qt = (question_type or "").lower()
    if "anomaly" in qt:
        return "anomaly"
    if qt.startswith("causal_"):
        return "causal"
    if qt.startswith("electricity_prediction") or qt.startswith("easy_stock"):
        return "forecasting"
    if qt.startswith("stock_rv_estimation"):
        return "risk_metric"
    if qt == "sharperatio":
        return "risk_metric"
    if (
        qt.startswith("var-")
        or qt.startswith("marketab-")
        or qt in {"stock_investment", "stock_ir_estimation"}
    ):
        return "decision"
    return "other"


# ---------------------------------------------------------------------------
# Series extraction (for fingerprint / memory keying)
# ---------------------------------------------------------------------------

# Pattern for bracketed numeric arrays embedded in TSAIA's text fields, e.g.
#   "...is: [10.01, 10.06, 10.0, ...]"
# Captures comma-separated numbers (with optional +/-/decimal/exponent). Allows
# whitespace and newlines inside the brackets but no nested brackets.
_NUMARRAY_RE = re.compile(r"\[([-+0-9eE.,\s]+)\]")


def _prepare_series(record: dict) -> dict[str, Any]:
    """Best-effort series extractor for TSAIA's text-embedded data.

    TSAIA does not ship per-record series in a structured field — the values
    are inlined in natural language inside ``data_str`` (analysis split) or
    ``data_info`` (MC split). We grep for ``[num, num, ...]`` patterns and
    treat each as one channel. Names default to ``series_0, series_1, ...``
    because the fingerprint extractor is shape-only and ignores names.

    Returns the ``load_data`` payload shape that ``compute_fingerprint``
    expects: ``{"channels": dict[str, list[float]], "timestamps": None,
    "meta": {...}}``. When no parseable arrays are found, ``channels`` is
    empty and the fingerprint will degenerate to zeros — caller can detect
    via ``not series["channels"]`` if it cares.

    Heuristic on purpose: this is a retrieval key, not a faithful
    reconstruction of the task data. The real values still live in the pkl
    files referenced by ``executor_variables`` / ``ground_truth_data`` for
    callers that need them.
    """
    if record.get("_split") == "mc":
        text = record.get("data_info") or ""
    else:
        text = record.get("data_str") or ""
    if not isinstance(text, str) or not text:
        return {"channels": {}, "timestamps": None, "meta": {}}

    channels: dict[str, list[float]] = {}
    for i, m in enumerate(_NUMARRAY_RE.finditer(text)):
        raw = m.group(1)
        # Sub-pattern: comma-split, ignore trailing commas/whitespace, drop
        # tokens that don't parse as float (defensive against e.g. `[A, B, C]`
        # accidentally matching when letters slipped through — unlikely with
        # the char class above, but keep the guard).
        vals: list[float] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                vals = []
                break
        # Skip arrays that are too short to fingerprint meaningfully.
        if len(vals) < 3:
            continue
        channels[f"series_{i}"] = vals

    meta = {
        "question_type": record.get("_question_type"),
        "category": record.get("_category"),
        "split": record.get("_split"),
    }
    return {"channels": channels, "timestamps": None, "meta": meta}


# Pattern for "next N {hours,days,trading hours,trading days}" — used to
# pull the forecast horizon out of MC prompts so we can stash it in meta
# for the quant tools.
_HORIZON_RE = re.compile(r"next\s+(\d+)\s+(?:trading\s+)?(hours?|days?)", re.IGNORECASE)


def _prepare_series_mc(record: dict) -> dict[str, Any]:
    """Load the actual MC executor_variables.pkl into MCP-ready channels.

    TSAIA's MC tasks ship the canonical numeric data in pickled DataFrames /
    ndarrays under ``executor_variables.pkl`` (the original TSAIA setup
    used a Python-REPL code-execution agent that ran against those
    variables). The text in ``data_info`` is a redundant inline copy that's
    often truncated or omits whole series (e.g. MarketAB has the asset
    inline but not the market index; SharpeRatio omits risk-free rates).

    Falls back to the regex-based ``_prepare_series`` when the pkl is not
    available so any code path that only needs a fingerprint still works.

    Channel naming convention (used by the prompt + tools):
      - VaR / SharpeRatio (per-option portfolios):
            ``A.stock_0``, ``A.stock_1``, ..., ``A.riskfree`` (if any),
            ``B.stock_0``, ..., ``C.stock_2``
      - MarketAB-{alpha,beta}: ``asset``, ``market``
    meta carries:
      - ``option_weights = {"A": [...], "B": [...], "C": [...]}`` for portfolio qts
      - ``horizon = int`` for VaR (parsed from the prompt's "next N hours")
      - ``question_type``, ``category``, ``split`` always
    """
    import os.path as _osp
    import pickle as _pickle

    qt = record.get("_question_type", "")
    base_meta = {
        "question_type": qt,
        "category": record.get("_category"),
        "split": record.get("_split"),
    }
    ev_path = record.get("executor_variables")
    if not isinstance(ev_path, str) or not ev_path.endswith(".pkl"):
        return _prepare_series(record)
    full = _osp.join(dataset_sources_dir, "TSAIA", ev_path)
    if not _osp.exists(full):
        return _prepare_series(record)
    try:
        with open(full, "rb") as f:
            ev = _pickle.load(f)
    except Exception:
        return _prepare_series(record)
    if not isinstance(ev, dict):
        return _prepare_series(record)

    def _to_list(obj) -> list[float] | None:
        try:
            arr = np.asarray(obj, dtype=float).ravel()
            return [float(v) for v in arr]
        except Exception:
            return None

    channels: dict[str, list[float]] = {}
    meta: dict[str, Any] = dict(base_meta)

    if qt.startswith("MarketAB"):
        for src_key, ch_name in (("VAL", "asset"), ("market_VAL", "market")):
            v = ev.get(src_key)
            if v is None:
                continue
            try:
                # DataFrame.values flatten to 1-D (single column).
                col = v.values.ravel() if hasattr(v, "values") else np.asarray(v).ravel()
                channels[ch_name] = [float(x) for x in col]
            except Exception:
                pass
    elif "VAL_A" in ev or "VAL_B" in ev or "VAL_C" in ev:
        # Portfolio-style task (VaR, SharpeRatio).
        option_weights: dict[str, list[float]] = {}
        for opt in ("A", "B", "C"):
            val_key, w_key, rf_key = f"VAL_{opt}", f"WEIGHTS_{opt}", f"RiskFreeRate_{opt}"
            val = ev.get(val_key)
            if val is None:
                continue
            try:
                mat = val.values if hasattr(val, "values") else np.asarray(val)
                if mat.ndim == 1:
                    mat = mat.reshape(-1, 1)
                for j in range(mat.shape[1]):
                    channels[f"{opt}.stock_{j}"] = [float(x) for x in mat[:, j]]
            except Exception:
                continue
            w = _to_list(ev.get(w_key))
            if w is not None:
                option_weights[opt] = w
            rf = _to_list(ev.get(rf_key))
            if rf is not None:
                channels[f"{opt}.riskfree"] = rf
        if option_weights:
            meta["option_weights"] = option_weights
    else:
        # Unknown shape — fall back to regex parser.
        return _prepare_series(record)

    # Extract forecast horizon (e.g. "next 11 trading hours") from prompt
    # so the quant tools can default to it without re-parsing the text.
    m = _HORIZON_RE.search(record.get("prompt") or "")
    if m:
        meta["horizon"] = int(m.group(1))
        meta["horizon_unit"] = m.group(2).rstrip("s").lower()

    if not channels:
        return _prepare_series(record)
    return {"channels": channels, "timestamps": None, "meta": meta}


# ---------------------------------------------------------------------------
# Text context for two-stage retrieval
# ---------------------------------------------------------------------------

# text-embedding-3-small caps inputs at 8192 tokens. TSAIA's
# climate_anomaly_large / similar can inline million-char stringified
# numpy arrays in `data_str`, so we hard-cap by chars (~3 chars/token
# for English; 12000 chars stays under ~4000 tokens with margin) at
# the helper output. The truncation is deterministic so train-time
# and test-time queries see the same prefix.
_MAX_EMBED_CHARS = 8000


def _tsaia_context_text(record: dict) -> str:
    """Join TSAIA's NL framing fields for text-embedding similarity.

    MC and analysis records carry different fields. We always include the
    semantic question_type label + prompt body; we then append the data
    description (data_str for analysis, data_info for MC) and the variant
    discriminator (output constraint for analysis, options list for MC).
    Mirrors `_cik_context_text` / `_tsrbench_context_text`. Output is
    truncated to `_MAX_EMBED_CHARS` to keep within the embedding API's
    8192-token input limit on outlier records.
    """
    parts: list[str] = []
    qt = (record.get("_question_type") or "").strip()
    if qt:
        parts.append(qt)
    prompt = (record.get("prompt") or "").strip()
    if prompt:
        parts.append(prompt)
    if record.get("_split") == "mc":
        data = (record.get("data_info") or "").strip()
        if data:
            parts.append(data)
        opts = record.get("options")
        if isinstance(opts, str):
            try:
                opts = json.loads(opts)
            except Exception:
                pass
        if isinstance(opts, list):
            joined = " ".join(str(o) for o in opts if str(o).strip())
            if joined:
                parts.append(joined)
        elif isinstance(opts, str) and opts.strip():
            parts.append(opts.strip())
    else:
        data = (record.get("data_str") or "").strip()
        if data:
            parts.append(data)
        constraint = (record.get("constraint") or "").strip()
        if constraint:
            parts.append(constraint)
    out = " | ".join(parts) or "(empty)"
    return out[:_MAX_EMBED_CHARS]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(dict(row))
    return out


def load_tsaia_records(subset: str = "all") -> list[dict]:
    """Load TSAIA records. ``subset`` is one of: 'all', 'analysis', 'mc',
    a category name, or a specific question_type."""
    analysis_path = osp.join(dataset_sources_dir, "TSAIA", "analysis_questions.csv")
    mc_path = osp.join(dataset_sources_dir, "TSAIA", "multiple_choice.csv")

    analysis_rows = _read_csv(analysis_path)
    mc_rows = _read_csv(mc_path)

    records: list[dict] = []
    for row in analysis_rows:
        qt = row.get("question_type", "")
        rec = dict(row)
        rec["_split"] = "analysis"
        rec["_question_type"] = qt
        rec["_category"] = tsaia_category_of(qt)
        rec["_task_id"] = f"analysis__{row.get('question_id', '')}"
        records.append(rec)
    for row in mc_rows:
        qt = row.get("question_type", "")
        rec = dict(row)
        rec["_split"] = "mc"
        rec["_question_type"] = qt
        rec["_category"] = tsaia_category_of(qt)
        rec["_task_id"] = f"mc__{row.get('question_id', '')}"
        records.append(rec)

    if subset is None or subset.lower() == "all":
        return records
    key = subset.lower()
    if key in {"analysis", "mc"}:
        return [r for r in records if r["_split"] == key]
    if key in {"anomaly", "causal", "forecasting", "decision", "risk_metric", "other"}:
        return [r for r in records if r["_category"] == key]
    # Treat as exact question_type
    out = [r for r in records if r["_question_type"] == subset]
    if not out:
        valid_cats = ["anomaly", "causal", "forecasting", "decision", "risk_metric"]
        valid_qts = sorted({r["_question_type"] for r in records})
        raise ValueError(
            f"Unknown TSAIA subset: {subset}. "
            f"Try one of categories={valid_cats} or splits=[analysis, mc] or a question_type from {valid_qts[:5]}..."
        )
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _try_parse_json(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(s)
        except Exception:
            continue
    return None


def _to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _flatten_numbers(obj) -> list[float]:
    out: list[float] = []
    if obj is None:
        return out
    if isinstance(obj, (int, float)):
        if not (isinstance(obj, float) and math.isnan(obj)):
            out.append(float(obj))
        return out
    if isinstance(obj, str):
        v = _to_float(obj)
        if v is not None:
            out.append(v)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_numbers(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_numbers(v))
        return out
    return out


def _score_analysis(record: dict, agent_text: str | None) -> dict[str, Any]:
    """Compute a metric bundle for one analysis row."""
    gt_raw = record.get("ground_truth_data", "")
    gt_obj = _try_parse_json(gt_raw)
    if gt_obj is None and gt_raw:
        gt_obj = gt_raw  # leave as string

    # Try to pull a structured prediction out of the agent text.
    pred_obj = None
    if agent_text:
        m = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", agent_text, re.DOTALL | re.IGNORECASE)
        if m:
            pred_obj = _try_parse_json(m.group(1)) or m.group(1).strip()
        if pred_obj is None:
            # Look for fenced JSON block
            blk = re.search(r"```(?:json)?\s*(.+?)```", agent_text, re.DOTALL)
            if blk:
                pred_obj = _try_parse_json(blk.group(1))
        if pred_obj is None:
            v = parse_number(agent_text)
            if v is not None:
                pred_obj = v

    metrics: dict[str, Any] = {"prediction": pred_obj, "gold_kind": "unknown"}

    gt_nums = _flatten_numbers(gt_obj)
    pred_nums = _flatten_numbers(pred_obj)

    if gt_nums and pred_nums and len(gt_nums) == len(pred_nums):
        metrics["gold_kind"] = "numeric_array_aligned"
        metrics["mae"] = mae(gt_nums, pred_nums)
        metrics["smape"] = smape(gt_nums, pred_nums)
    elif gt_nums and pred_nums:
        # Length mismatch: compare min length common prefix as a best effort,
        # but also report length delta for visibility.
        n = min(len(gt_nums), len(pred_nums))
        metrics["gold_kind"] = "numeric_array_mismatch"
        metrics["mae"] = mae(gt_nums[:n], pred_nums[:n])
        metrics["smape"] = smape(gt_nums[:n], pred_nums[:n])
        metrics["len_delta"] = len(pred_nums) - len(gt_nums)
    elif isinstance(gt_obj, str) and isinstance(pred_obj, str):
        metrics["gold_kind"] = "string"
        metrics["exact_match"] = bool(gt_obj.strip().lower() == pred_obj.strip().lower())
    else:
        metrics["gold_kind"] = "unscored"
        metrics["mae"] = float("nan")

    return metrics


def _score_mc(record: dict, agent_text: str | None) -> dict[str, Any]:
    opts = record.get("options")
    if isinstance(opts, str):
        opts = _try_parse_json(opts) or []
    n_opts = len(opts) if isinstance(opts, list) else 4
    valid = tuple(chr(ord("A") + i) for i in range(min(n_opts, 5)))
    pred = parse_mc_letter(agent_text or "", valid_letters=valid)

    gold = record.get("answer", "")
    if isinstance(gold, str):
        gold = gold.strip().upper()
        gold = gold[0] if gold else None
    return {"prediction": pred, "gold": gold, "correct": (pred is not None and pred == gold)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def run_tsaia(
    agent: TimeClaw,
    *,
    mode: str = "full",
    train_ratio: float = 0.5,
    split_seed: int = 2026,
    k_neighbors: int = 3,
    retrieve_same_family_only: bool = False,
    text_filter_size: int = 0,
    num_workers: int = 4,
    ratio: float = 1.0,
    seed: int = 42,
    subset: str = "all",
) -> str:
    records = load_tsaia_records(subset=subset)
    group_key_fn = lambda r: r["_question_type"]
    sampled = subsample_groups(records, group_key_fn=group_key_fn, ratio=ratio, seed=seed)
    selected, split_stats = select_split(
        sampled, group_key_fn, mode=mode, train_ratio=train_ratio, split_seed=split_seed,
    )

    # k_neighbors only affects test mode (retrieval); tag the dir with it so
    # the k=0 baseline and k=3 memory run don't share a folder name.
    run_dir = make_run_dir(
        "tsaia", agent.model, mode,
        ratio=ratio,
        subset=subset,
        k=k_neighbors if mode == "test" else None,
    )
    writer = EvalWriter(
        run_dir,
        RunConfig(
            benchmark="tsaia",
            model=agent.model,
            mode=mode,
            ratio=ratio,
            seed=seed,
            train_ratio=train_ratio if mode != "full" else 0.0,
            split_seed=split_seed,
            num_workers=num_workers,
            timestamp=osp.basename(run_dir),
            extra={
                "subset": subset,
                "n_records": len(selected),
                "n_total": len(records),
                **split_stats,
            },
        ),
    )

    bank = None
    if mode in ("train", "test"):
        bank = MemoryBank.for_split(
            benchmark="tsaia",
            model=agent.model,
            split_seed=split_seed,
            train_ratio=train_ratio,
            ratio=ratio,
        )
        if mode == "test" and bank.is_empty:
            print(
                f"[tsaia] mode=test: memory bank at {bank.bank_dir} is empty — "
                "retrieval will be a no-op. Run --mode train first to populate it.",
                flush=True,
            )

    async def _task(record: dict) -> dict:
        # Build a fingerprint from the regex-extracted series inline in
        # data_str / data_info. TSAIA doesn't preload via MCP, but the
        # fingerprint is computed harness-side and is independent of the
        # agent's tool access.
        series = _prepare_series(record)
        refs = None
        if mode == "test" and bank is not None and not bank.is_empty:
            fp = compute_fingerprint(series)
            text_query_emb = None
            if text_filter_size > 0:
                text_query_emb = embed_text(_tsaia_context_text(record))
            neighbors = bank.retrieve_topk(
                fp,
                k=k_neighbors,
                family_filter=record["_question_type"] if retrieve_same_family_only else None,
                text_query_embedding=text_query_emb,
                text_filter_size=text_filter_size if text_filter_size > 0 else None,
            )
            refs = [summarize_trajectory(n.record) for n in neighbors]

        # MC GT is a single letter; analysis GT is the structured
        # ``ground_truth_data`` (dict / list / scalar) — the suffix helper
        # JSON-dumps the latter automatically. Analysis rows sometimes
        # store GT as a path to an external .pkl rather than inline data;
        # showing that path to the agent would be actively misleading, so
        # we drop the GT injection in that case.
        if record["_split"] == "mc":
            gt_for_prompt = record.get("answer") if mode == "train" else None
            tool_series = _prepare_series_mc(record)
            prompt = build_tsaia_mc_prompt_with_tools(
                record, ground_truth=gt_for_prompt, memory_references=refs,
            )
        else:
            gt_raw = record.get("ground_truth_data") if mode == "train" else None
            gt_for_prompt = gt_raw
            if isinstance(gt_raw, str) and (
                gt_raw.endswith(".pkl") or "external_data/" in gt_raw
            ):
                gt_for_prompt = None
            tool_series = None
            prompt = build_tsaia_analysis_prompt(
                record, ground_truth=gt_for_prompt, memory_references=refs,
            )
        result = await call_agent_once(agent, prompt, series=tool_series)
        text = result.text or ""
        if record["_split"] == "mc":
            scored = _score_mc(record, text)
        else:
            scored = _score_analysis(record, text)
        line = {
            "task_id": record["_task_id"],
            "split": record["_split"],
            "question_type": record["_question_type"],
            "category": record["_category"],
            "scored": scored,
            "raw_text": text,
            "token_info": result.token_info,
            "elapsed_s": result.elapsed,
            "error": result.error,
        }
        await writer.write_prediction(line, trajectory=result.raw_response)
        if mode == "train" and bank is not None and result.error is None:
            # Build the bank record. Analysis rows with pkl-path GTs surface
            # the path string verbatim (cheap and faithful — the resolved
            # value will be re-pickled at load time anyway).
            if record["_split"] == "mc":
                ground_truth = record.get("answer")
            else:
                ground_truth = record.get("ground_truth_data")
            log_record: dict[str, Any] = {
                "task_id": record["_task_id"],
                "family_key": record["_question_type"],
                "fingerprint": compute_fingerprint(series),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model_answer": result.text,
                "trajectory": result.raw_response,
                "model": agent.model,
            }
            # Embed the NL context once at log time so test-mode retrievals
            # can use the bank without re-embedding the training set every
            # time. Only spend the API call when text retrieval is enabled.
            if text_filter_size > 0:
                log_record["text_embedded"] = _tsaia_context_text(record)
                log_record["text_embedding"] = embed_text(
                    log_record["text_embedded"]
                ).tolist()
            await bank.log(log_record)
        return line

    results = await run_tasks_concurrent(selected, _task, num_workers=num_workers, desc="TSAIA")

    # Aggregation
    by_qt: dict[str, list[dict]] = defaultdict(list)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_qt[r["question_type"]].append(r)
        by_cat[r["category"]].append(r)

    def _block(items: list[dict]) -> dict:
        mc_items = [it for it in items if it["split"] == "mc"]
        an_items = [it for it in items if it["split"] == "analysis"]
        block: dict[str, Any] = {
            "n_tasks": len(items),
            "n_api_errors": sum(1 for it in items if it["error"]),
            "mean_elapsed_s": (
                sum(it["elapsed_s"] for it in items) / len(items) if items else float("nan")
            ),
            "total_tokens": sum(
                (it["token_info"] or {}).get("total_tokens", 0) for it in items
            ),
        }
        if mc_items:
            preds = [it["scored"]["prediction"] for it in mc_items]
            golds = [it["scored"]["gold"] for it in mc_items]
            block["mc_accuracy"] = accuracy(preds, golds)
            block["mc_n"] = len(mc_items)
        if an_items:
            maes = [it["scored"].get("mae") for it in an_items]
            smps = [it["scored"].get("smape") for it in an_items]
            ems = [it["scored"].get("exact_match") for it in an_items if "exact_match" in it["scored"]]
            block["analysis_mae"] = aggregate_numeric(maes)
            block["analysis_smape"] = aggregate_numeric(smps)
            if ems:
                block["analysis_exact_match"] = float(sum(ems) / len(ems))
            block["analysis_n"] = len(an_items)
            block["analysis_n_scored"] = sum(
                1
                for it in an_items
                if it["scored"].get("gold_kind", "unscored") != "unscored"
            )
        return block

    summary = {
        "per_question_type": {qt: _block(items) for qt, items in by_qt.items()},
        "per_category": {cat: _block(items) for cat, items in by_cat.items()},
        "overall": _block(results),
        "mode": mode,
    }
    if bank is not None:
        summary["memory_bank"] = {
            "dir": str(bank.bank_dir),
            "n_records": len(bank),
            "k_neighbors": k_neighbors if mode == "test" else None,
        }
    writer.finalize_summary(summary)
    return run_dir
