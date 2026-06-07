"""TSRBench evaluator.

Tasks are split into 12 sub-tasks across 4 categories. All tasks are MCQ
(closed-set options), so we score with exact-match accuracy.
"""

from __future__ import annotations

import json
import numbers
import os.path as osp
from collections import defaultdict
from typing import Any

import pandas as pd

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
from timeclaw.evaluation.metrics import accuracy
from timeclaw.evaluation.parsers import parse_mc_letter
from timeclaw.evaluation.prompts import build_tsrbench_prompt_with_tools
from timeclaw.memory import (
    MemoryBank,
    compute_fingerprint,
    embed_text,
    summarize_trajectory,
)
from timeclaw.utils.path import dataset_sources_dir


# ---------------------------------------------------------------------------
# Sub-task → category mapping
# ---------------------------------------------------------------------------

TASK_PATHS: dict[str, str] = {
    "perception": "perception/perception.jsonl",
    "event_prediction": "prediction/event_prediction.jsonl",
    "time_series_forecasting": "prediction/time_series_forecasting.jsonl",
    "qualitative_decision": "decision/qualitative_decision.jsonl",
    "quantitative_decision": "decision/quantitative_decision.jsonl",
    "abductive_reasoning": "reasoning/abductive_reasoning.jsonl",
    "causal_reasoning": "reasoning/causal_reasoning.jsonl",
    "deductive_reasoning": "reasoning/deductive_reasoning.jsonl",
    "etiological_reasoning": "reasoning/etiological_reasoning.jsonl",
    "inductive_reasoning": "reasoning/inductive_reasoning.jsonl",
    "numerical_reasoning": "reasoning/numerical_reasoning.jsonl",
    "temporal_relation_reasoning": "reasoning/temporal_relation_reasoning.jsonl",
}

CATEGORY_TO_TASKS: dict[str, list[str]] = {
    "perception": ["perception"],
    "prediction": ["event_prediction", "time_series_forecasting"],
    "decision": ["qualitative_decision", "quantitative_decision"],
    "reasoning": [
        "abductive_reasoning",
        "causal_reasoning",
        "deductive_reasoning",
        "etiological_reasoning",
        "inductive_reasoning",
        "numerical_reasoning",
        "temporal_relation_reasoning",
    ],
}

TASK_TO_CATEGORY: dict[str, str] = {
    t: cat for cat, tasks in CATEGORY_TO_TASKS.items() for t in tasks
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tsrbench_records(subset: str = "all") -> list[dict]:
    """Load TSRBench records, possibly filtered to a category or a sub-task."""
    if subset is None or subset.lower() == "all":
        selected = list(TASK_PATHS.keys())
    else:
        key = subset.lower()
        if key in CATEGORY_TO_TASKS:
            selected = CATEGORY_TO_TASKS[key]
        elif key in TASK_PATHS:
            selected = [key]
        else:
            valid = sorted(set(TASK_PATHS) | set(CATEGORY_TO_TASKS))
            raise ValueError(f"Unknown TSRBench subset: {subset}. Valid: {valid}")

    records: list[dict] = []
    for task_name in selected:
        path = osp.join(dataset_sources_dir, "TSRBench", TASK_PATHS[task_name])
        df = pd.read_json(path, lines=True)
        for idx, row in df.iterrows():
            rec = row.to_dict()
            # abductive_reasoning records nest question/choices/answer inside a
            # `multiple_choice_question` dict and also carry `numerical_time_series`
            # instead of `timeseries`. Flatten so the shared prompt builder + scorer
            # works without subtask-specific code paths downstream.
            if task_name == "abductive_reasoning":
                mcq = rec.get("multiple_choice_question")
                if isinstance(mcq, dict):
                    rec.setdefault("question", mcq.get("question", ""))
                    rec.setdefault("choices", mcq.get("choices"))
                    rec.setdefault("answer", mcq.get("answer"))
                if "timeseries" not in rec and "numerical_time_series" in rec:
                    rec["timeseries"] = rec["numerical_time_series"]
            rec["_subtask"] = task_name
            rec["_category"] = TASK_TO_CATEGORY[task_name]
            rec["_task_id"] = f"{task_name}__{idx}"
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Series payload for MCP load_data
# ---------------------------------------------------------------------------

def _coerce_floats(seq: Any) -> list[float]:
    """Convert any iterable of numeric values to a fresh list[float]."""
    return [float(v) for v in seq]


def prepare_series(record: dict) -> dict[str, Any]:
    """Normalize a TSRBench record's time series into the load_data payload.

    Output shape (consumed by ``timeclaw.tools.server.load_data``):

        {"channels": {name: list[float]},
         "timestamps": list[str] | None,
         "meta": {subtask, category, domain, task}}

    Handles all 12 subtasks of TSRBench, dispatched on the actual shape of
    ``record["timeseries"]`` rather than the subtask name:

    - dict-of-dicts (abductive_reasoning's `numerical_time_series` flattened
      under load_tsrbench_records): each ``feature.split`` becomes one channel
    - list-of-lists: multi-channel; if a channel's first element is a string
      it's treated as the timestamp axis (TSRBench's temporal_relation pattern
      where ``timeseries = [[ts_str, ...], [val, ...]]``)
    - flat list of numbers: a single channel named via ``name_of_series`` if
      present else ``series_0``

    Channel names come from ``record["name_of_series"]`` where available, so
    e.g. deductive_reasoning surfaces "Choice A: x(t)" / "Choice B: x(t)" /
    ... and qualitative_decision keeps its ECG lead names.
    """
    ts = record.get("timeseries")
    names = record.get("name_of_series")
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except Exception:
            names = None
    if not isinstance(names, list):
        names = None

    channels: dict[str, list[float]] = {}
    timestamps: list[str] | None = None

    if isinstance(ts, dict):
        # abductive_reasoning: {feature: {history: [...], future: [...]}}
        for feature, splits in ts.items():
            key_base = str(feature)
            if isinstance(splits, dict):
                for split_name, vals in splits.items():
                    if isinstance(vals, list):
                        channels[f"{key_base}.{split_name}"] = _coerce_floats(vals)
            elif isinstance(splits, list):
                channels[key_base] = _coerce_floats(splits)
    elif isinstance(ts, list) and ts and isinstance(ts[0], list):
        # Multi-channel (incl. temporal_relation's [timestamps, values] shape)
        for i, channel in enumerate(ts):
            if not isinstance(channel, list) or not channel:
                continue
            first = channel[0]
            if isinstance(first, str):
                timestamps = [str(v) for v in channel]
                continue
            name = names[i] if names and i < len(names) else f"channel_{i}"
            channels[str(name)] = _coerce_floats(channel)
    elif isinstance(ts, list) and ts and isinstance(ts[0], numbers.Real):
        # Flat single-channel
        name = names[0] if names else "series_0"
        channels[str(name)] = _coerce_floats(ts)

    meta = {
        "subtask": record.get("_subtask"),
        "category": record.get("_category"),
        "domain": record.get("domain"),
        "task": record.get("task"),
    }
    return {"channels": channels, "timestamps": timestamps, "meta": meta}


# ---------------------------------------------------------------------------
# Text context for two-stage retrieval
# ---------------------------------------------------------------------------

# text-embedding-3-small caps inputs at 8192 tokens. Cap helper output
# so outlier records (e.g. time_series_forecasting questions that
# embed long numeric arrays) can't overflow the API.
_MAX_EMBED_CHARS = 8000


def _tsrbench_context_text(record: dict) -> str:
    """Join TSRBench's NL framing fields for text-embedding similarity.

    Mirrors `_cik_context_text`: concatenate the bits that identify what the
    task is asking, not the raw series data. ``choices`` is intentionally
    omitted — several subtasks (time_series_forecasting, causal_reasoning)
    encode numeric arrays / matrices there which dominate the embedding
    with noise instead of semantic signal. Output truncated to
    ``_MAX_EMBED_CHARS``.
    """
    def _s(v: Any) -> str:
        """Coerce a possibly-NaN/None/non-string field to a stripped string."""
        if v is None or not isinstance(v, str):
            # pd.read_json yields NaN (float) for missing fields; treat as empty.
            if isinstance(v, float) and v != v:  # NaN
                return ""
            return "" if v is None else str(v).strip()
        return v.strip()

    parts: list[str] = []
    sub = _s(record.get("_subtask"))
    if sub:
        parts.append(sub)
    for f in ("domain", "task"):
        v = _s(record.get(f))
        if v:
            parts.append(v)
    names = record.get("name_of_series")
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except Exception:
            pass
    if isinstance(names, list):
        joined = ", ".join(str(n) for n in names if str(n).strip())
        if joined:
            parts.append(joined)
    elif isinstance(names, str) and names.strip():
        parts.append(names.strip())
    question = _s(record.get("question"))
    if question:
        parts.append(question)
    out = " | ".join(parts) or "(empty)"
    return out[:_MAX_EMBED_CHARS]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _gold_letter(record: dict) -> str | None:
    """Normalize ground-truth answer to a single uppercase letter."""
    ans = record.get("answer")
    if ans is None:
        return None
    s = str(ans).strip().upper()
    if not s:
        return None
    # Some entries are "A)" or "A. ..." — take the leading letter
    for ch in s:
        if ch.isalpha():
            return ch
    return None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def run_tsrbench(
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
    records = load_tsrbench_records(subset=subset)
    group_key_fn = lambda r: r["_subtask"]
    sampled = subsample_groups(records, group_key_fn=group_key_fn, ratio=ratio, seed=seed)
    selected, split_stats = select_split(
        sampled, group_key_fn, mode=mode, train_ratio=train_ratio, split_seed=split_seed,
    )

    # k_neighbors only affects test mode (retrieval); tag the dir with it so
    # the k=0 baseline and k=3 memory run don't share a folder name.
    run_dir = make_run_dir(
        "tsrbench", agent.model, mode,
        ratio=ratio,
        subset=subset,
        k=k_neighbors if mode == "test" else None,
    )
    writer = EvalWriter(
        run_dir,
        RunConfig(
            benchmark="tsrbench",
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

    prompt_builder = build_tsrbench_prompt_with_tools

    # Train mode opens the bank for writing; test mode opens it for read-only
    # retrieval. Full mode skips it entirely. The bank path is a pure function
    # of (model, split_seed, train_ratio, ratio) so a train run and a later
    # test run with matching args land on the same directory automatically;
    # ratio is part of the key to prevent test-set / bank contamination
    # across ratios (see bank_path_for docstring).
    bank = None
    if mode in ("train", "test"):
        bank = MemoryBank.for_split(
            benchmark="tsrbench",
            model=agent.model,
            split_seed=split_seed,
            train_ratio=train_ratio,
            ratio=ratio,
        )
        if mode == "test" and bank.is_empty:
            print(
                f"[tsrbench] mode=test: memory bank at {bank.bank_dir} is empty — "
                "retrieval will be a no-op. Run --mode train first to populate it.",
                flush=True,
            )

    async def _task(record: dict) -> dict:
        # Training mode feeds the raw answer field (which may be "A", "A. ...",
        # or a free-form number/text depending on subtask) into the prompt
        # suffix verbatim. _gold_letter below is the parsed letter used for
        # MCQ scoring; both paths coexist intentionally.
        raw_answer = record.get("answer")
        gt_for_prompt = raw_answer if mode == "train" else None
        series = prepare_series(record)
        # Test mode: compute the same deterministic fingerprint used at log
        # time and pull k-nearest references from the bank.
        refs = None
        if mode == "test" and bank is not None and not bank.is_empty:
            fp = compute_fingerprint(series)
            text_query_emb = None
            if text_filter_size > 0:
                text_query_emb = embed_text(_tsrbench_context_text(record))
            neighbors = bank.retrieve_topk(
                fp,
                k=k_neighbors,
                family_filter=record["_subtask"] if retrieve_same_family_only else None,
                text_query_embedding=text_query_emb,
                text_filter_size=text_filter_size if text_filter_size > 0 else None,
            )
            refs = [summarize_trajectory(n.record) for n in neighbors]
        prompt = prompt_builder(record, ground_truth=gt_for_prompt, memory_references=refs)
        result = await call_agent_once(agent, prompt, series=series)
        # Determine valid letters from #options
        choices = record.get("choices")
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except Exception:
                choices = None
        n_opts = len(choices) if isinstance(choices, list) else 5
        valid = tuple(chr(ord("A") + i) for i in range(min(n_opts, 5)))
        pred = parse_mc_letter(result.text or "", valid_letters=valid)
        gold = _gold_letter(record)
        line = {
            "task_id": record["_task_id"],
            "subtask": record["_subtask"],
            "category": record["_category"],
            "domain": record.get("domain"),
            "type": record.get("type"),
            "gold": gold,
            "prediction": pred,
            "correct": (pred is not None and gold is not None and pred == gold),
            "raw_text": result.text,
            "token_info": result.token_info,
            "elapsed_s": result.elapsed,
            "error": result.error,
        }
        await writer.write_prediction(line, trajectory=result.raw_response)
        if mode == "train" and bank is not None and result.error is None:
            log_record: dict[str, Any] = {
                "task_id": record["_task_id"],
                "family_key": record["_subtask"],
                "fingerprint": compute_fingerprint(series),
                "prompt": prompt,
                "ground_truth": raw_answer,
                "model_answer": result.text,
                "model_answer_parsed": pred,
                "trajectory": result.raw_response,
                "model": agent.model,
            }
            # Embed the NL context once at log time so test-mode retrievals
            # can use the bank without re-embedding the training set every
            # time. Only spend the API call when text retrieval is enabled.
            if text_filter_size > 0:
                log_record["text_embedded"] = _tsrbench_context_text(record)
                log_record["text_embedding"] = embed_text(
                    log_record["text_embedded"]
                ).tolist()
            await bank.log(log_record)
        return line

    results = await run_tasks_concurrent(selected, _task, num_workers=num_workers, desc="TSRBench")

    # Aggregate by subtask and category
    summary: dict[str, Any] = {"per_subtask": {}, "per_category": {}, "overall": {}}

    by_sub: dict[str, list[dict]] = defaultdict(list)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_sub[r["subtask"]].append(r)
        by_cat[r["category"]].append(r)

    def _block(items: list[dict]) -> dict:
        preds = [it["prediction"] for it in items]
        golds = [it["gold"] for it in items]
        return {
            "n_tasks": len(items),
            "n_parsed": sum(1 for p in preds if p is not None),
            "n_api_errors": sum(1 for it in items if it["error"]),
            "accuracy": accuracy(preds, golds),
            "mean_elapsed_s": (
                sum(it["elapsed_s"] for it in items) / len(items) if items else float("nan")
            ),
            "total_tokens": sum(
                (it["token_info"] or {}).get("total_tokens", 0) for it in items
            ),
        }

    for k, items in by_sub.items():
        summary["per_subtask"][k] = _block(items)
    for k, items in by_cat.items():
        summary["per_category"][k] = _block(items)
    summary["overall"] = _block(results)
    summary["mode"] = mode
    if bank is not None:
        summary["memory_bank"] = {
            "dir": str(bank.bank_dir),
            "n_records": len(bank),
            "k_neighbors": k_neighbors if mode == "test" else None,
        }

    writer.finalize_summary(summary)
    return run_dir
