"""OpenAI text-embedding-3-small wrapper for memory retrieval.

Used by the harness to convert a task's NL context (background + scenario
+ constraints for CiK, prompt body for others) into a 1536-dim vector for
two-stage retrieval (text first, fingerprint within).

Why text-embedding-3-small specifically:
  - 1536 dims, plenty for our scale (few hundred records per bank)
  - $0.02 / 1M tokens → full CiK bank (~50K tokens) costs ~$0.001
  - Outputs are L2-normalized so cosine == dot product
  - No new dependency: ``openai`` is already in the env

The embedding is computed once at log time, stored alongside the record
in bank.jsonl, and re-used at every test retrieval. Test-time queries
re-embed the test record's context once per query.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from openai import BadRequestError, OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# text-embedding-3-small caps input at 8192 tokens. Callers are
# expected to pre-truncate, but BPE on number-dense / non-English text
# can push the token count well above char-based estimates, so we
# halve-and-retry once on overflow as a last-resort guard.
_MAX_TOKEN_RETRY = 2

# Lazy-init so importing this module doesn't require an API key
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def embed_text(text: str | Iterable[str]) -> np.ndarray:
    """Embed one or many strings via text-embedding-3-small.

    Returns shape ``(D,)`` for a single string input and ``(N, D)`` for a
    list. Empty / None strings are replaced with " " so the API doesn't
    reject them; the resulting embedding is essentially noise but the
    pipeline stays robust.

    On a token-overflow ``BadRequestError`` the inputs are halved by
    character count and retried (up to ``_MAX_TOKEN_RETRY`` rounds) so
    one outlier record can't kill a whole training run.
    """
    if isinstance(text, str):
        texts = [text]
        single = True
    else:
        texts = list(text)
        single = False
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    safe = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
    for attempt in range(_MAX_TOKEN_RETRY + 1):
        try:
            resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=safe)
            break
        except BadRequestError as e:
            if "maximum input length" not in str(e) or attempt == _MAX_TOKEN_RETRY:
                raise
            safe = [t[: max(1, len(t) // 2)] for t in safe]
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return arr[0] if single else arr
