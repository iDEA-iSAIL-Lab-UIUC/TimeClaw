"""Extract structured answers from free-form LLM agent responses.

All parsers are best-effort: they return ``None`` (or NaN-filled placeholders)
when extraction fails so the eval pipeline can record the failure without
crashing.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# MCQ letter
# ---------------------------------------------------------------------------

_ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>\s*([A-Z])\s*<\s*/\s*answer\s*>", re.IGNORECASE)
_ANSWER_IS_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\(?\s*([A-Z])\b", re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r"\b([A-Z])\b")


def parse_mc_letter(text: str, valid_letters: tuple[str, ...] = ("A", "B", "C", "D", "E")) -> str | None:
    """Extract a single MCQ letter, preferring tagged answers > 'answer is X' > last bare letter."""
    if not text:
        return None
    m = _ANSWER_TAG_RE.search(text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()
    m = _ANSWER_IS_RE.search(text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()
    # Fallback: last bare uppercase letter that is in valid_letters.
    candidates = [lt for lt in _BARE_LETTER_RE.findall(text) if lt in valid_letters]
    if candidates:
        return candidates[-1]
    return None


# ---------------------------------------------------------------------------
# Single number
# ---------------------------------------------------------------------------

_NUM_TAG_RE = re.compile(r"<\s*answer\s*>\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*<\s*/\s*answer\s*>", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_number(text: str) -> float | None:
    """Extract a single scalar number from agent output.

    Search order: <answer>N</answer> > JSON {"answer": N} > last number in text.
    """
    if not text:
        return None
    m = _NUM_TAG_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    for obj_str in _JSON_OBJ_RE.findall(text):
        try:
            obj = json.loads(obj_str)
        except Exception:
            continue
        for key in ("answer", "value", "prediction", "result"):
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
    nums = _NUMBER_RE.findall(text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Forecast: 1D array or quantile dict
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _normalize_length(vals: list[float], expected: int) -> list[float] | None:
    """Force a numeric list to exactly ``expected`` length.

    Too long → keep the first ``expected`` (agent miscount, off-by-N, or
    extra trailing zeros).
    Too short → pad with the last value to reach ``expected`` (a common
    failure mode where the agent emits only one channel of a multi-channel
    forecast, or just stops early).
    Empty input → ``None``: we have nothing to extend from.
    """
    if not vals:
        return None
    if len(vals) >= expected:
        return vals[:expected]
    return vals + [vals[-1]] * (expected - len(vals))


def _normalize_samples(samples: list[list[float]], expected: int) -> list[list[float]] | None:
    """Same pad/truncate logic but on rows of a quantile sample matrix."""
    if not samples:
        return None
    if len(samples) >= expected:
        return samples[:expected]
    return samples + [samples[-1]] * (expected - len(samples))


def parse_forecast(text: str, expected_length: int) -> dict[str, Any] | None:
    """Extract a forecast from agent output, normalizing length to ``expected_length``.

    Length policy: if a structurally-valid forecast is found but the value
    count differs from ``expected_length``, we keep the first
    ``expected_length`` (when too long) or pad with the last value (when
    too short). One CiK record (LocaleInfoHalfDay solar) had a perfectly
    shaped forecast off by one (69 vs 68); strict equality kept failing it
    while pad/truncate recovers the substantive forecast. The same policy
    applies to ``samples`` rows in a quantile forecast.

    Returns:
        - ``{"kind": "point", "values": [N values]}`` if a point forecast
          is found (after pad/truncate to ``expected_length``).
        - ``{"kind": "quantile", "levels": [...], "samples": [N rows]}``
          for a quantile structure.
        - ``None`` if no structurally-valid forecast at all (empty or
          all-non-numeric).
    """
    if not text:
        return None
    # Try JSON in fenced code block first
    json_blobs: list[str] = []
    json_blobs.extend(_FENCED_JSON_RE.findall(text))
    # Also raw arrays
    raw_arr = re.search(r"\[\s*(?:[-+]?\d[\d\.\-+eE]*\s*,\s*){2,}[-+]?\d[\d\.\-+eE]*\s*\]", text)
    if raw_arr:
        json_blobs.append(raw_arr.group(0))

    for blob in json_blobs:
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        # quantile structure
        if isinstance(obj, dict):
            if "forecast" in obj and isinstance(obj["forecast"], list):
                vals = obj["forecast"]
                if vals and all(isinstance(v, (int, float)) for v in vals):
                    norm = _normalize_length([float(v) for v in vals], expected_length)
                    if norm is not None:
                        return {"kind": "point", "values": norm}
            if "levels" in obj and "samples" in obj:
                lv = obj["levels"]
                samp = obj["samples"]
                if (
                    isinstance(lv, list)
                    and isinstance(samp, list)
                    and samp
                    and all(isinstance(s, list) and len(s) == len(lv) for s in samp)
                ):
                    norm = _normalize_samples(
                        [[float(v) for v in s] for s in samp], expected_length
                    )
                    if norm is not None:
                        return {
                            "kind": "quantile",
                            "levels": [float(q) for q in lv],
                            "samples": norm,
                        }
        elif isinstance(obj, list):
            if obj and all(isinstance(v, (int, float)) for v in obj):
                norm = _normalize_length([float(v) for v in obj], expected_length)
                if norm is not None:
                    return {"kind": "point", "values": norm}

    # No fenced or raw-array forecast at all. Earlier versions had a "last
    # resort" that swept up the last N numbers anywhere in the text, but
    # that caused parse-time hallucination when the agent refused to
    # forecast and instead produced a clarification with stray numbers
    # (timestamps, indices, etc.) — those got scored as if they were
    # predictions. Fail-silent here; the runner counts as parse_fail and
    # scoring upstream uses a bounded penalty.
    return None
