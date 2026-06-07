"""LangChain response parsing and serialization helpers."""

from typing import Any

from timeclaw.utils.model_provider import (
    is_anthropic_model,
    is_gemini_model,
    is_openai_model,
)


def serialize_langchain_response(raw_response: Any) -> dict:
    """Convert a LangChain agent response into a JSON-serializable dict.

    LangChain message objects are first turned into plain dicts using
    ``.dict()`` / ``.model_dump()`` when available, with a string fallback.
    """
    try:
        if isinstance(raw_response, dict) and "messages" in raw_response:
            messages: list[dict] = []
            for msg in raw_response["messages"]:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    messages.append(msg.dict())
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"content": str(msg), "type": type(msg).__name__})
            return {"messages": messages}

        if hasattr(raw_response, "model_dump"):
            return raw_response.model_dump()
        if hasattr(raw_response, "dict"):
            return raw_response.dict()
        if isinstance(raw_response, dict):
            return dict(raw_response)
        return {"raw": str(raw_response)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Serialization failed: {e}", "raw": str(raw_response)}


def _coerce_message_text(content: Any) -> str:
    """Normalize a LangChain message ``content`` into a plain string.

    OpenAI messages always carry ``content`` as ``str``. Gemini and Claude
    return a ``list[dict]`` of content parts whenever the response has more
    than one part (e.g. text + reasoning, or text + tool_call), e.g.
    ``[{"type": "text", "text": "..."}]``. Downstream parsers expect a
    string, so we concatenate the ``text`` fields of any text parts.
    Non-text parts (tool_use blocks, etc.) are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return "" if content is None else str(content)


def parse_agent_response(
    model: str, response: dict[str, Any]
) -> tuple[str, int, int, int, int]:
    """Extract ``(text, completion, prompt, total, reasoning)`` token counts.

    Token counts are summed across every AI message in the response, which
    for tool-using agents covers the full tool-call loop (one AI message per
    LLM round).
    """
    messages = response["messages"]
    last = messages[-1]
    text = _coerce_message_text(last.content)

    if is_openai_model(model):
        comp = prompt_t = total = reason = 0
        for msg in messages:
            meta = getattr(msg, "response_metadata", None)
            if not isinstance(meta, dict):
                continue
            usage = meta.get("token_usage")
            if not isinstance(usage, dict):
                continue
            comp += usage.get("completion_tokens", 0) or 0
            prompt_t += usage.get("prompt_tokens", 0) or 0
            total += usage.get("total_tokens", 0) or 0
            details = usage.get("completion_tokens_details") or {}
            reason += details.get("reasoning_tokens", 0) or 0
        return text, comp, prompt_t, total, reason
    if is_gemini_model(model) or is_anthropic_model(model):
        # LangChain standardizes ``usage_metadata`` across these providers to
        # ``{input_tokens, output_tokens, total_tokens, ...}``, with
        # extended-thinking / reasoning tokens (when present) reported under
        # ``output_token_details.reasoning``.
        comp = prompt_t = total = reason = 0
        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if not isinstance(usage, dict):
                continue
            comp += usage.get("output_tokens", 0) or 0
            prompt_t += usage.get("input_tokens", 0) or 0
            total += usage.get("total_tokens", 0) or 0
            details = usage.get("output_token_details") or {}
            reason += details.get("reasoning", 0) or 0
        return text, comp, prompt_t, total, reason
    raise ValueError(f"Model {model!r} not supported by parser.")