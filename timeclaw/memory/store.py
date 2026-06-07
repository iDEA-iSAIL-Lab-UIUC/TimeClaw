"""jsonl-backed memory bank with in-memory L2 nearest-neighbor retrieval.

A MemoryBank is a per-(benchmark, model, split) persistent store of training
trajectories used to inform test-time retrieval. The on-disk layout is
intentionally simple:

    {root}/{benchmark}/{safe_model}/seed{split_seed}_tr{tr_pct:03d}/
      bank.jsonl       # one record per line (canonical source of truth)
      meta.json        # schema + feature compatibility info

jsonl is the canonical store; the numpy fingerprint matrix and per-feature
scaler are lazily derived on the first retrieval call and cached in memory.
This means a training run only ever does append-only writes (fast, safe
under concurrent workers) and a test run does a one-time bank load + many
queries.

Retrieval uses Euclidean distance on z-scored fingerprints. Z-score uses
the bank's per-feature mean/std so different scales (e.g. log_length vs
ACF lag-1) don't let one feature dominate the metric. The scaler is
rebuilt on every fresh bank load; it is not persisted because rebuilding
costs ~0ms for a few-thousand-row bank.

At our scale (per-benchmark bank typically <= a few thousand records,
fingerprint dim ~20) a numpy broadcast L2 search beats FAISS in
operational complexity and is ~1ms per query. The retrieval path is
isolated to ``_ensure_index`` + ``retrieve_topk`` so swapping to FAISS
later is a single-class-method change.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from timeclaw.memory.fingerprint import FEATURE_NAMES, FINGERPRINT_DIM
from timeclaw.memory.text_embed import EMBEDDING_DIM


SCHEMA_VERSION = 1


def _feature_names_hash() -> str:
    """SHA-256 of the current FEATURE_NAMES tuple — guards against silent drift."""
    h = hashlib.sha256("|".join(FEATURE_NAMES).encode("utf-8")).hexdigest()
    return h[:16]


def _safe_path_segment(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def bank_path_for(
    benchmark: str,
    model: str,
    split_seed: int,
    train_ratio: float,
    ratio: float,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Canonical bank directory for a (benchmark, model, split, ratio) tuple.

    Pure function of its inputs; same arguments always return the same path
    so a training run and a later test run land on the same bank.

    ``ratio`` (the within-family subsampling ratio applied BEFORE the
    train/test split) is part of the key because two runs that share
    (model, split_seed, train_ratio) but differ on ratio can have their test
    splits overlap with the OTHER run's train split, which would let
    retrieval pull a record's own ground truth from the bank. Segregating
    banks per ratio eliminates that contamination.
    """
    root_path = Path(root) if root is not None else Path("memory_banks")
    tr_pct = int(round(max(0.0, min(1.0, train_ratio)) * 100))
    r_pct = int(round(max(0.0, min(1.0, ratio)) * 100))
    return (
        root_path
        / _safe_path_segment(benchmark)
        / _safe_path_segment(model)
        / f"seed{split_seed}_tr{tr_pct:03d}_r{r_pct:03d}"
    )


@dataclass
class Neighbor:
    """One retrieval result. Wraps the full stored record + its distance."""
    record: dict
    distance: float


class MemoryBank:
    """Append-only jsonl store of training trajectories + L2 NN retrieval.

    Construct via the classmethod ``for_split(...)`` for the canonical layout,
    or directly with a ``bank_dir`` path if you need a one-off bank.
    """

    def __init__(self, bank_dir: str | os.PathLike[str]):
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.bank_dir / "bank.jsonl"
        self.meta_path = self.bank_dir / "meta.json"

        self._records: list[dict] = []
        self._seen_task_ids: set[str] = set()
        # Lazily-built retrieval cache; cleared on every log() so the next
        # retrieve_topk() rebuilds. At our scale rebuilding is essentially
        # free, and this avoids stale-index bugs.
        self._fp_matrix: np.ndarray | None = None     # (N, D) raw
        self._scaler_mu: np.ndarray | None = None     # (D,)
        self._scaler_sd: np.ndarray | None = None     # (D,)
        # Text embedding cache for two-stage retrieval. None entries mark
        # records logged without an embedding (older banks); retrieve_topk
        # will treat them as ineligible for the text-filter stage.
        self._text_emb_matrix: np.ndarray | None = None       # (N, 1536)
        self._text_emb_mask: np.ndarray | None = None         # (N,) bool, True where present

        # Single write lock for cross-worker append safety. Created lazily
        # since asyncio.Lock binds to the current event loop.
        self._write_lock: asyncio.Lock | None = None

        self._load_existing()

    # ------------------------------------------------------------------ paths

    @classmethod
    def for_split(
        cls,
        benchmark: str,
        model: str,
        split_seed: int,
        train_ratio: float,
        ratio: float,
        root: str | os.PathLike[str] | None = None,
    ) -> "MemoryBank":
        return cls(
            bank_path_for(benchmark, model, split_seed, train_ratio, ratio, root)
        )

    # ------------------------------------------------------------------ load

    def _load_existing(self) -> None:
        """Load existing jsonl content into memory; validate meta compatibility."""
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    f"bank at {self.bank_dir} has schema_version={meta.get('schema_version')}, "
                    f"current code expects {SCHEMA_VERSION}; rebuild the bank."
                )
            if meta.get("feature_names_hash") != _feature_names_hash():
                raise RuntimeError(
                    f"bank at {self.bank_dir} was built with a different FEATURE_NAMES set "
                    f"(stored hash {meta.get('feature_names_hash')!r}, "
                    f"current hash {_feature_names_hash()!r}); rebuild the bank to retrieve safely."
                )
            if meta.get("fingerprint_dim") != FINGERPRINT_DIM:
                raise RuntimeError(
                    f"bank at {self.bank_dir} has fingerprint_dim={meta.get('fingerprint_dim')}, "
                    f"current code uses {FINGERPRINT_DIM}; rebuild the bank."
                )

        if not self.jsonl_path.exists():
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Skip a partially-written tail line from a crashed run.
                    continue
                fp = rec.get("fingerprint")
                if not isinstance(fp, list) or len(fp) != FINGERPRINT_DIM:
                    continue
                self._records.append(rec)
                tid = rec.get("task_id")
                if tid is not None:
                    self._seen_task_ids.add(tid)

    def _write_meta(self) -> None:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint_dim": FINGERPRINT_DIM,
            "feature_names": list(FEATURE_NAMES),
            "feature_names_hash": _feature_names_hash(),
            "bank_dir": str(self.bank_dir).replace(os.sep, "/"),
            "n_records": len(self._records),
            "last_updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        if not self.meta_path.exists():
            meta["created_at"] = meta["last_updated_at"]
        else:
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                meta["created_at"] = prev.get("created_at", meta["last_updated_at"])
            except (json.JSONDecodeError, OSError):
                meta["created_at"] = meta["last_updated_at"]
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------ write

    async def log(self, record: dict) -> bool:
        """Append a training record. Returns True if appended, False if skipped.

        Skips silently when ``record["task_id"]`` is already in the bank so an
        interrupted training run can be resumed by simply re-running it; the
        same call site doesn't need to track which records succeeded.

        ``record`` must include ``task_id``, ``family_key``, ``fingerprint``
        (a 1-D numpy array or list of FINGERPRINT_DIM floats). Other fields
        (prompt, ground_truth, trajectory, model, ...) are written through
        verbatim — schema is intentionally loose.
        """
        task_id = record.get("task_id")
        if task_id is None:
            raise ValueError("memory bank record must have 'task_id'")
        if task_id in self._seen_task_ids:
            return False

        fp = record.get("fingerprint")
        if isinstance(fp, np.ndarray):
            fp_list = [float(x) for x in fp.ravel().tolist()]
        elif isinstance(fp, (list, tuple)):
            fp_list = [float(x) for x in fp]
        else:
            raise TypeError(f"record['fingerprint'] must be ndarray or list, got {type(fp)}")
        if len(fp_list) != FINGERPRINT_DIM:
            raise ValueError(
                f"fingerprint must have length {FINGERPRINT_DIM}, got {len(fp_list)}"
            )

        out = dict(record)
        out["fingerprint"] = fp_list
        out.setdefault("logged_at", datetime.now(tz=timezone.utc).isoformat())

        line = json.dumps(out, default=str, ensure_ascii=False) + "\n"

        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        async with self._write_lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            self._records.append(out)
            self._seen_task_ids.add(task_id)
            # Invalidate the retrieval cache; rebuilt lazily on next read.
            self._fp_matrix = None
            self._scaler_mu = None
            self._scaler_sd = None
            self._text_emb_matrix = None
            self._text_emb_mask = None
            self._write_meta()
        return True

    # ------------------------------------------------------------------ read

    def _ensure_index(self) -> None:
        """Build the (N, D) matrix + per-feature z-score scaler from current records."""
        if self._fp_matrix is not None:
            return
        if not self._records:
            self._fp_matrix = np.zeros((0, FINGERPRINT_DIM), dtype=np.float32)
            self._scaler_mu = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
            self._scaler_sd = np.ones(FINGERPRINT_DIM, dtype=np.float32)
            return
        mat = np.vstack([np.asarray(r["fingerprint"], dtype=np.float32) for r in self._records])
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        # Guard zero-variance features so retrieval doesn't divide by 0;
        # they become identically-zero columns post-scale (no contribution).
        sd = np.where(sd < 1e-9, 1.0, sd)
        self._fp_matrix = mat
        self._scaler_mu = mu.astype(np.float32)
        self._scaler_sd = sd.astype(np.float32)

    def _ensure_text_index(self) -> None:
        """Build the (N, 1536) text-embedding matrix + presence mask.

        Records that were logged without a text_embedding (older banks, or
        benchmarks that don't supply one) get zero rows in the matrix and
        False in the mask. The text-filter stage of retrieve_topk skips
        records where the mask is False.
        """
        if self._text_emb_matrix is not None:
            return
        n = len(self._records)
        if n == 0:
            self._text_emb_matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._text_emb_mask = np.zeros(0, dtype=bool)
            return
        mat = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, r in enumerate(self._records):
            emb = r.get("text_embedding")
            if isinstance(emb, list) and len(emb) == EMBEDDING_DIM:
                mat[i] = np.asarray(emb, dtype=np.float32)
                mask[i] = True
        self._text_emb_matrix = mat
        self._text_emb_mask = mask

    def retrieve_topk(
        self,
        fingerprint: np.ndarray | list[float],
        k: int = 3,
        exclude_task_id: str | None = None,
        family_filter: str | None = None,
        text_query_embedding: np.ndarray | list[float] | None = None,
        text_filter_size: int | None = None,
    ) -> list[Neighbor]:
        """Return up to k nearest training records.

        Single-stage (default): L2 on z-scored fingerprint features over the
        whole bank, with optional ``exclude_task_id`` / ``family_filter`` to
        narrow the candidate pool.

        Two-stage (when both ``text_query_embedding`` and
        ``text_filter_size`` > 0 are given and the bank has stored
        embeddings): stage 1 picks the top ``text_filter_size`` records by
        cosine on the text embedding (background+scenario+constraints), and
        stage 2 ranks the survivors by fingerprint L2. This is the
        production retrieval path; the text stage handles the
        "find the right family / sibling task" problem (which fingerprint
        can't solve when CiK's framing variants share identical past_time)
        and the fingerprint stage discriminates by series shape within
        those candidates. Records without a stored text_embedding are not
        eligible for the text stage and will be dropped from the candidate
        pool — i.e. enabling the text filter on a bank without embeddings
        effectively yields an empty result.

        Returns may be shorter than ``k`` when filters eliminate
        candidates.
        """
        self._ensure_index()
        n = self._fp_matrix.shape[0]
        if n == 0:
            return []

        q = np.asarray(fingerprint, dtype=np.float32).ravel()
        if q.shape != (FINGERPRINT_DIM,):
            raise ValueError(
                f"query fingerprint must have shape ({FINGERPRINT_DIM},), got {q.shape}"
            )

        # Stage 1 (optional): text-cosine top-K_text candidate filter
        candidate_mask: np.ndarray | None = None
        if (
            text_query_embedding is not None
            and text_filter_size is not None
            and text_filter_size > 0
        ):
            self._ensure_text_index()
            if self._text_emb_mask.any():
                q_text = np.asarray(text_query_embedding, dtype=np.float32).ravel()
                if q_text.shape != (EMBEDDING_DIM,):
                    raise ValueError(
                        f"text_query_embedding must have shape ({EMBEDDING_DIM},), got {q_text.shape}"
                    )
                # Cosine via dot; OpenAI embeddings are L2-normalized.
                text_sims = self._text_emb_matrix @ q_text
                # Records without an embedding score 0; mask them out so they
                # never make the top-K_text cut even if K_text is large.
                text_sims = np.where(self._text_emb_mask, text_sims, -np.inf)
                kt = min(text_filter_size, int(self._text_emb_mask.sum()))
                if kt > 0:
                    top_text_idx = np.argpartition(-text_sims, kth=kt - 1)[:kt]
                    candidate_mask = np.zeros(n, dtype=bool)
                    candidate_mask[top_text_idx] = True

        # Z-score both query and stored features by the bank scaler.
        q_z = (q - self._scaler_mu) / self._scaler_sd
        m_z = (self._fp_matrix - self._scaler_mu) / self._scaler_sd
        dists = np.linalg.norm(m_z - q_z, axis=1)

        if exclude_task_id is not None:
            for i, r in enumerate(self._records):
                if r.get("task_id") == exclude_task_id:
                    dists[i] = np.inf
        if family_filter is not None:
            for i, r in enumerate(self._records):
                if r.get("family_key") != family_filter:
                    dists[i] = np.inf
        if candidate_mask is not None:
            dists = np.where(candidate_mask, dists, np.inf)

        kk = min(k, n)
        # argpartition picks the kk smallest, then sort just that slice.
        part = np.argpartition(dists, kth=kk - 1)[:kk]
        ordered = part[np.argsort(dists[part])]
        # Drop entries pushed to inf (filter eliminated them).
        return [
            Neighbor(record=self._records[int(i)], distance=float(dists[int(i)]))
            for i in ordered
            if np.isfinite(dists[int(i)])
        ]

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[dict]:
        return iter(self._records)

    @property
    def is_empty(self) -> bool:
        return not self._records

    def has_task(self, task_id: str) -> bool:
        return task_id in self._seen_task_ids
