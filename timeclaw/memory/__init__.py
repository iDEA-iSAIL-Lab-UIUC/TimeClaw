"""TimeClaw memory layer.

- ``fingerprint``: deterministic ~20-dim numerical descriptor of a series,
  used as the retrieval key.
- ``store``: append-only jsonl bank of training trajectories + L2 NN
  retrieval over z-scored fingerprints. One bank per (benchmark, model,
  split_seed, train_ratio, ratio).
- ``text_embed``: text-embedding-3-small wrapper for the NL-context
  pre-filter that wraps fingerprint retrieval.
- ``summarize``: render a stored trajectory back into a compact
  prompt-injectable reference.
"""

from timeclaw.memory.fingerprint import (
    FEATURE_NAMES,
    FINGERPRINT_DIM,
    compute_fingerprint,
)
from timeclaw.memory.store import MemoryBank, Neighbor, bank_path_for
from timeclaw.memory.summarize import summarize_trajectory
from timeclaw.memory.text_embed import EMBEDDING_DIM, EMBEDDING_MODEL, embed_text

__all__ = [
    "FEATURE_NAMES",
    "FINGERPRINT_DIM",
    "compute_fingerprint",
    "MemoryBank",
    "Neighbor",
    "bank_path_for",
    "summarize_trajectory",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "embed_text",
]
