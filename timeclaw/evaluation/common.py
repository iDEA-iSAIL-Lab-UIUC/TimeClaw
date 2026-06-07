"""Shared utilities for the benchmark eval harness:

- ``subsample_groups``: deterministic ratio-based subsampling within groups.
- ``run_tasks_concurrent``: async runner with a worker semaphore + tqdm.
- ``EvalWriter``: streams ``predictions.jsonl`` and dumps summary at the end.
- ``call_agent_once``: thin wrapper around ``traced_ainvoke`` that catches errors.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import os.path as osp
import random
import secrets
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Sequence

from tqdm.asyncio import tqdm_asyncio

from timeclaw.agents import TimeClaw, traced_ainvoke
from timeclaw.utils.path import results_dir


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_groups(
    records: Sequence[dict],
    group_key_fn: Callable[[dict], str],
    ratio: float,
    seed: int,
    min_per_group: int = 1,
) -> list[dict]:
    """Subsample ``records`` to ``ratio`` per group, with a deterministic seed.

    Within each group identified by ``group_key_fn(record)``, draw
    ``max(min_per_group, ceil(ratio * group_size))`` items using a
    ``random.Random(seed)`` shuffle. When ``ratio >= 1.0`` we keep everything.
    Returns the records in their original order (filtered).
    """
    if ratio >= 1.0:
        return list(records)
    if ratio <= 0.0:
        return []

    # Group while preserving original indices.
    groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        groups.setdefault(group_key_fn(rec), []).append(idx)

    rng = random.Random(seed)
    keep: set[int] = set()
    for key in sorted(groups.keys()):  # sort for determinism across runs
        idxs = list(groups[key])
        rng.shuffle(idxs)
        n_keep = max(min_per_group, math.ceil(ratio * len(idxs)))
        n_keep = min(n_keep, len(idxs))
        keep.update(idxs[:n_keep])

    return [records[i] for i in sorted(keep)]


def split_train_test(
    records: Sequence[dict],
    group_key_fn: Callable[[dict], str],
    train_ratio: float,
    seed: int,
    min_train_per_group: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Within each group, partition ``records`` into ``(train, test)``.

    For each group of size ``n``, train gets
    ``min(n, max(min_train_per_group, ceil(train_ratio * n)))`` records and
    test gets the rest. ``min_train_per_group`` guarantees every family is
    seen at training time (so memory retrieval at test time always has at
    least one same-family exemplar to look up). A group of size 1 puts its
    single record in train and contributes nothing to test.

    The split is deterministic for a fixed ``seed`` and uses a shuffle within
    each group, so it is independent of record order. Both returned lists
    preserve the original ``records`` order.
    """
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")

    groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        groups.setdefault(group_key_fn(rec), []).append(idx)

    rng = random.Random(seed)
    train_idx: set[int] = set()
    test_idx: set[int] = set()
    for key in sorted(groups.keys()):  # sort for cross-run determinism
        idxs = list(groups[key])
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(min_train_per_group, math.ceil(train_ratio * n))
        n_train = min(n_train, n)
        train_idx.update(idxs[:n_train])
        test_idx.update(idxs[n_train:])

    train = [records[i] for i in sorted(train_idx)]
    test = [records[i] for i in sorted(test_idx)]
    return train, test


def select_split(
    sampled: Sequence[dict],
    group_key_fn: Callable[[dict], str],
    mode: str,
    train_ratio: float,
    split_seed: int,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Apply ``--mode`` to a subsampled record list.

    Returns ``(selected, stats)``. In ``full`` mode the entire subsampled set
    is returned and no split is performed (``n_train`` / ``n_test`` in stats
    are 0). In ``train`` / ``test`` mode ``selected`` is the corresponding
    half of a deterministic within-family split keyed by ``split_seed``.

    When ``verbose`` (default), prints a per-family breakdown so the operator
    can spot coverage gaps before any API calls.
    """
    if mode == "full":
        if verbose:
            from collections import Counter
            c = Counter(group_key_fn(r) for r in sampled)
            print(f"[split] mode=full, subsampled={len(sampled)} across {len(c)} families:")
            for k in sorted(c):
                print(f"  {k:<36s} {c[k]:>4d}")
        return list(sampled), {"n_subsampled": len(sampled), "n_train": 0, "n_test": 0}
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be one of 'full','train','test', got {mode!r}")
    train, test = split_train_test(sampled, group_key_fn, train_ratio, split_seed)
    selected = train if mode == "train" else test
    if verbose:
        from collections import Counter
        c_sub = Counter(group_key_fn(r) for r in sampled)
        c_tr = Counter(group_key_fn(r) for r in train)
        c_te = Counter(group_key_fn(r) for r in test)
        n_test_zero = sum(1 for k in c_sub if c_te[k] == 0)
        print(
            f"[split] mode={mode}, subsampled={len(sampled)}, "
            f"train={len(train)} test={len(test)} (train_ratio={train_ratio}, "
            f"split_seed={split_seed})"
        )
        print(f"  per-family breakdown ({len(c_sub)} families, "
              f"{n_test_zero} have 0 test records):")
        print(f"  {'family':<36s} {'sub':>4s} {'train':>6s} {'test':>5s}")
        for k in sorted(c_sub):
            marker = "  <-- 0 test" if c_te[k] == 0 else ""
            print(f"  {k:<36s} {c_sub[k]:>4d} {c_tr[k]:>6d} {c_te[k]:>5d}{marker}")
    return selected, {
        "n_subsampled": len(sampled),
        "n_train": len(train),
        "n_test": len(test),
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    benchmark: str
    model: str
    mode: str            # "full" | "train" | "test"
    ratio: float
    seed: int
    train_ratio: float   # 0.0 when mode == "full"
    split_seed: int      # ignored when mode == "full"
    num_workers: int
    timestamp: str
    extra: dict = field(default_factory=dict)


class EvalWriter:
    """Streams predictions.jsonl + trajectories.jsonl and finalizes summary.

    Each call to ``write_prediction`` writes one line to ``predictions.jsonl``
    AND one line to ``trajectories.jsonl`` under the same lock, so row i of
    one file always corresponds to row i of the other (same task_id, same
    completion order). The trajectory row carries the full LangChain message
    chain (HumanMessage prompt, AIMessage(s) with reasoning + tool_calls,
    ToolMessage(s) with tool outputs).
    """

    def __init__(self, run_dir: str, run_config: RunConfig):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.predictions_path = osp.join(self.run_dir, "predictions.jsonl")
        self.trajectories_path = osp.join(self.run_dir, "trajectories.jsonl")
        self.summary_path = osp.join(self.run_dir, "summary.json")
        self.config_path = osp.join(self.run_dir, "run_config.json")
        self._predictions_fh = open(self.predictions_path, "w", encoding="utf-8")
        self._trajectories_fh = open(self.trajectories_path, "w", encoding="utf-8")
        self._lock = asyncio.Lock()
        self._run_config = run_config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(run_config), f, indent=2)

    async def write_prediction(self, record: dict, trajectory: Any = None) -> None:
        """Append one row to predictions.jsonl + one row to trajectories.jsonl.

        Both writes happen under the same lock so the files stay row-aligned
        even under concurrent workers. ``trajectory`` is the dict from
        ``serialize_langchain_response`` (or ``None`` on API error); the
        trajectory row always carries ``task_id`` so a row can still be
        joined back if the alignment is ever broken (e.g. partial run).
        """
        task_id = record.get("task_id")
        traj_row = {"task_id": task_id, "trajectory": trajectory}
        async with self._lock:
            self._predictions_fh.write(json.dumps(record, default=str) + "\n")
            self._predictions_fh.flush()
            self._trajectories_fh.write(json.dumps(traj_row, default=str, ensure_ascii=False) + "\n")
            self._trajectories_fh.flush()

    def finalize_summary(self, summary: dict) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        self._predictions_fh.close()
        self._trajectories_fh.close()


def make_run_dir(
    benchmark: str,
    model: str,
    mode: str,
    ratio: float,
    subset: str | None = None,
    k: int | None = None,
    root: str | None = None,
) -> str:
    """Create a fresh, never-existing run directory.

    Layout: ``{root}/{benchmark}/{model}_{mode}[_{subset}][_k{k}]_r{ratio_pct}_{date}-{time}[_{nonce}]/``.
    Descriptive tokens lead so ``ls`` groups runs by config; the timestamp at
    the end keeps within-config runs chronologically ordered. ``subset`` is
    appended when it is set and not the default (``"all"`` / ``None``). ``k``
    (the retrieval neighbor count) is appended only for ``test`` mode, so
    the k=0 and k=3 runs don't collide. ``ratio`` is always appended as
    ``r{NNN}`` so runs at different ratios are visually distinct.
    """
    root = root or results_dir
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    safe_mode = mode.replace("/", "_").replace(":", "_")
    parts = [safe_model, safe_mode]
    if subset and subset != "all":
        parts.append(subset.replace("/", "_").replace(":", "_"))
    if k is not None:
        parts.append(f"k{k}")
    r_pct = int(round(max(0.0, min(1.0, ratio)) * 100))
    parts.append(f"r{r_pct:03d}")
    prefix = "_".join(parts)

    # Try without a nonce first; only collide-retry if the same-second name
    # already exists.
    base = osp.join(root, benchmark, f"{prefix}_{ts}")
    try:
        os.makedirs(base, exist_ok=False)
        return base
    except FileExistsError:
        pass
    for _ in range(8):
        nonce = secrets.token_hex(2)
        run_dir = osp.join(root, benchmark, f"{prefix}_{ts}_{nonce}")
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run dir under {root}/{benchmark}")


# ---------------------------------------------------------------------------
# Agent call wrapper
# ---------------------------------------------------------------------------

@dataclass
class AgentCallResult:
    text: str | None
    token_info: dict
    elapsed: float
    raw_response: dict | None
    error: str | None


async def call_agent_once(
    agent: TimeClaw,
    prompt: str,
    series: dict[str, Any] | None = None,
) -> AgentCallResult:
    """Invoke the agent once. Never raises: wraps exceptions into AgentCallResult.

    Model id is read from ``agent.model``. ``series`` is an optional payload
    preloaded into the agent's MCP server before the LLM call.
    """
    t0 = time.perf_counter()
    try:
        text, token_info, elapsed, raw = await traced_ainvoke(agent, prompt, series=series)
        return AgentCallResult(
            text=text, token_info=token_info, elapsed=elapsed, raw_response=raw, error=None
        )
    except Exception as exc:  # noqa: BLE001
        return AgentCallResult(
            text=None,
            token_info={},
            elapsed=time.perf_counter() - t0,
            raw_response=None,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[:2000]}",
        )


# ---------------------------------------------------------------------------
# Concurrent runner
# ---------------------------------------------------------------------------

async def run_tasks_concurrent(
    items: Sequence[Any],
    task_fn: Callable[[Any], Awaitable[dict]],
    num_workers: int,
    desc: str = "tasks",
) -> list[dict]:
    """Run ``task_fn`` over ``items`` with at most ``num_workers`` concurrent calls."""
    sem = asyncio.Semaphore(max(1, num_workers))

    async def _wrapped(item):
        async with sem:
            return await task_fn(item)

    coros = [_wrapped(it) for it in items]
    results = await tqdm_asyncio.gather(*coros, desc=desc, total=len(coros))
    return results
