"""TimeClaw agent: LLM + in-memory MCP server with time-series analysis tools."""

import asyncio
import os
import time
from typing import Any

from fastmcp import Client as FastMCPClient
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools

from timeclaw.tools.server import make_timeclaw_mcp_server
from timeclaw.utils.model_provider import (
    is_anthropic_model,
    is_gemini_model,
    is_openai_model,
)
from timeclaw.utils.parse import parse_agent_response, serialize_langchain_response


def initialize_model(model: str):
    """Return a LangChain-compatible model handle for ``model``.

    OpenAI models are referenced by name (``create_agent`` resolves them);
    Gemini goes through ``ChatGoogleGenerativeAI``; Claude goes through
    ``ChatAnthropic`` (reads ``ANTHROPIC_API_KEY`` and optional
    ``ANTHROPIC_BASE_URL``).
    """
    if is_openai_model(model):
        return model
    if is_gemini_model(model):
        return ChatGoogleGenerativeAI(model=model)
    if is_anthropic_model(model):
        max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192"))
        max_retries = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "10"))
        return ChatAnthropic(
            model=model, max_tokens=max_tokens, max_retries=max_retries,
        )
    raise ValueError(f"Model {model!r} is not supported.")


async def traced_ainvoke(
    agent: Any,
    prompt: str,
    series: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], float, dict[str, Any]]:
    """Invoke an agent and capture latency, token usage, and a serialized trace."""
    start = time.perf_counter()
    raw_response = await agent.invoke(prompt, series=series)
    elapsed = time.perf_counter() - start

    result_text, completion_tokens, prompt_tokens, total_tokens, reasoning_tokens = (
        parse_agent_response(agent.model, raw_response)
    )
    token_info = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    raw_response_dict = serialize_langchain_response(raw_response)
    return result_text, token_info, elapsed, raw_response_dict


class _TimeClawSlot:
    """One (in-memory FastMCP server, MCP client, LangChain agent) triple.

    A slot is the per-worker unit of concurrency. The slot's MCP server is the
    only one the slot's agent can reach, so loaded data and any other
    per-task state stay isolated even when multiple tasks run concurrently.
    """

    def __init__(self, model: str):
        self.model = model
        self._server, self._harness_load_data = make_timeclaw_mcp_server()
        self._client = FastMCPClient(self._server)
        self._client_entered = False
        self._lc_agent: Any = None

    async def open(self) -> None:
        await self._client.__aenter__()
        self._client_entered = True
        lc_tools = await load_mcp_tools(self._client.session)
        self._lc_agent = create_agent(
            model=initialize_model(self.model), tools=lc_tools
        )

    async def close(self) -> None:
        if self._client_entered:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client_entered = False

    async def load_data(self, series: dict[str, Any]) -> None:
        """Push a load_data payload to this slot's MCP server.

        ``series`` is a dict produced by per-benchmark normalizers: keys are
        ``channels`` (dict[str, list[float]]), ``timestamps`` (optional list
        of str), ``meta`` (optional dict).

        Bypasses the MCP transport on purpose: the slot's series state is
        mutated directly via the harness callable returned by
        ``make_timeclaw_mcp_server``. The agent's tool list does NOT include
        load_data, so a confused LLM cannot accidentally re-trigger this.
        """
        self._harness_load_data(
            series.get("channels") or {},
            series.get("timestamps"),
            series.get("meta") or {},
        )

    async def invoke(self, prompt: str) -> Any:
        return await self._lc_agent.ainvoke({"messages": prompt})


class TimeClaw:
    """LLM agent backed by a pool of in-memory MCP servers, one per worker.

    ``pool_size`` should match the eval harness ``num_workers`` so every
    concurrent task gets a dedicated slot. Slots are created lazily on the
    first ``invoke`` call (so construction stays sync and cheap); call
    ``close()`` when done to release MCP client sessions.
    """

    def __init__(self, model: str, pool_size: int = 1):
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.model = model
        self.pool_size = pool_size
        self._slots: list[_TimeClawSlot] = []
        self._available: asyncio.Queue | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_init(self) -> None:
        if self._available is not None:
            return
        async with self._init_lock:
            if self._available is not None:
                return
            slots = [_TimeClawSlot(self.model) for _ in range(self.pool_size)]
            for s in slots:
                await s.open()
            q: asyncio.Queue = asyncio.Queue()
            for s in slots:
                q.put_nowait(s)
            self._slots = slots
            self._available = q

    async def invoke(self, prompt: str, series: dict[str, Any] | None = None) -> Any:
        """Borrow a slot, optionally preload its MCP server with task data,
        then run the LLM. The preload + LLM invocation share one slot for the
        whole call, so tool calls during the LLM round see the same loaded
        data even under concurrent borrows by other tasks.
        """
        await self._ensure_init()
        slot = await self._available.get()
        try:
            if series is not None:
                await slot.load_data(series)
            return await slot.invoke(prompt)
        finally:
            self._available.put_nowait(slot)

    async def close(self) -> None:
        for s in self._slots:
            await s.close()
        self._slots = []
        self._available = None
