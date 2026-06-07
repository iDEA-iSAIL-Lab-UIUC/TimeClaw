"""Compact text summaries of stored bank records for test-time injection.

A trained bank record carries the full LangChain message chain (HumanMessage
prompt, alternating AIMessage(tool_calls=[...]) + ToolMessage(content=...)
pairs, then a final AIMessage with the answer). For retrieval-time injection
we don't want to dump the entire chain back into the test agent's prompt —
just the analytic spine: which tools the trainer called, the args, and an
abbreviated view of each tool's response, plus the final answer.

``summarize_trajectory`` produces a short multi-line block per record that
the prompt prefix helper concatenates into a "REFERENCES FROM PRIOR
TRAINING" section.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_RESPONSE_TRUNCATE = 200      # per-tool-response visible chars
_FINAL_ANSWER_TRUNCATE = 120
_REASONING_TRUNCATE = 500          # context_to_forecast block max chars

# Match the training-mode <context_to_forecast> block. Tolerant of case,
# whitespace, and missing close tag (we still grab whatever the agent
# emitted after the opening tag, up to the next answer tag or end of msg).
_CTF_RE = re.compile(
    r"<\s*context_to_forecast\s*>(.*?)(?:<\s*/\s*context_to_forecast\s*>|<\s*answer\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_tool_text(content: Any) -> str:
    """Pull the inner ``text`` field out of a ToolMessage payload.

    FastMCP wraps tool responses as ``[{"type": "text", "text": "..."}]``
    (sometimes nested) and ToolMessage stores that envelope as a JSON
    string. Fall through to the raw string if parsing fails so the caller
    still gets something useful.
    """
    if isinstance(content, list):
        try:
            return content[0].get("text", str(content))
        except (AttributeError, IndexError):
            return str(content)
    if not isinstance(content, str):
        return str(content)
    s = content.strip()
    if s.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return obj[0].get("text", s)
        except json.JSONDecodeError:
            pass
    return s


def _format_args(args: Any) -> str:
    """Compact one-line representation of a tool_call's args dict."""
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _extract_context_to_forecast(msgs: list[dict]) -> str | None:
    """Pull the <context_to_forecast> block from the trainer's final AI message.

    The block is emitted by the training prompt's suffix and is intended
    to capture WHY the GT looks the way it does, in transferable terms
    (e.g. "scenario describes a heat wave → expect a multi-fold spike at
    the stated time"). We extract it so the test-time injection can
    surface this reasoning to the test agent verbatim.
    """
    for m in reversed(msgs):
        if m.get("type") != "ai":
            continue
        if m.get("tool_calls"):
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content:
            continue
        match = _CTF_RE.search(content)
        if match:
            text = match.group(1).strip()
            if text:
                return text
        # Don't break — the agent might emit a tool_calls=[] but no answer
        # message; keep looking at earlier ai messages just in case.
    return None


def summarize_trajectory(record: dict) -> str:
    """Produce a compact, test-injection-ready summary of one bank record.

    The summary is a small fixed shape:

        [family=<family_key>]
          context→forecast: <reasoning block, if the trainer emitted one>
          tool_name(args) → <truncated_response>
          tool_name(args) → <truncated_response>
          ...

    The trainer's GT answer and final-answer text are deliberately OMITTED:
    they cause strong answer-anchoring at test time and consistently
    degrade accuracy in our ablations. The current shape exposes only the
    analytic spine + the optional context→forecast explanation block.

    The context→forecast block, when present, is the trainer's own
    explanation of how the textual context drives the answer's shape. It
    is the most directly transferable piece across same-family tasks and
    is surfaced first in the summary so the test-time agent reads it
    before the tool-call details.

    Truncation is bounded so a top-k=3 retrieval typically adds well
    under 2.5 KB to the prompt.
    """
    family = record.get("family_key", "?")
    msgs = (record.get("trajectory") or {}).get("messages", [])

    header = f"[family={family}]"
    lines: list[str] = [header]

    reasoning = _extract_context_to_forecast(msgs)
    if reasoning:
        if len(reasoning) > _REASONING_TRUNCATE:
            reasoning = reasoning[:_REASONING_TRUNCATE] + "..."
        # Collapse newlines so the block stays one logical line inside the
        # composite Reference N block.
        reasoning = re.sub(r"\s+", " ", reasoning).strip()
        lines.append(f"  context→forecast: {reasoning}")

    pending: tuple[str, str] | None = None  # (tool_name, args_str)

    for m in msgs:
        mtype = m.get("type")
        if mtype == "human":
            continue
        if mtype == "ai":
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                name = tc.get("name", "?")
                args_str = _format_args(tc.get("args"))
                pending = (name, args_str)
            # Final-answer AIMessages are deliberately not surfaced — see
            # the docstring above.
        elif mtype == "tool" and pending is not None:
            text = _extract_tool_text(m.get("content", ""))
            if len(text) > _TOOL_RESPONSE_TRUNCATE:
                text = text[:_TOOL_RESPONSE_TRUNCATE] + "..."
            text = text.replace("\n", " ")
            name, args_str = pending
            lines.append(f"  {name}({args_str}) → {text}")
            pending = None

    return "\n".join(lines)
