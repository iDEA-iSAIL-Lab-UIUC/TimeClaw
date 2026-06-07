"""Prompt builders for each benchmark.

Each builder returns a single ``str`` prompt that is fed to the TimeClaw
agent. Prompts are deliberately strict about output format so the parsers
downstream have a high success rate.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _retrieval_prefix(memory_references: list[str] | None) -> str:
    """Render top-k retrieved bank summaries as a prompt-prefix block.

    Returns "" when ``memory_references`` is None or empty so callers can
    unconditionally concatenate it. The block is fenced with ASCII markers
    that are unlikely to appear in the question text, so the model can
    visually separate priors from the actual task.
    """
    if not memory_references:
        return ""
    body = "\n\n".join(
        f"Reference {i + 1}:\n{ref}" for i, ref in enumerate(memory_references)
    )
    return (
        "=== REFERENCES FROM PRIOR TRAINING ===\n"
        "Below are similar prior tasks (same task family where possible) and how\n"
        "they were solved — which tools were called, the key numbers returned,\n"
        "and the final answer. Use them as a guide for the analytic process. The\n"
        "correct answer for the current task may differ; do not copy blindly.\n\n"
        f"{body}\n"
        "=== END REFERENCES ===\n\n"
    )


def _training_suffix(ground_truth: Any) -> str:
    """Append a GT-aware training instruction to a prompt.

    When ``ground_truth`` is None this returns the empty string so callers
    can unconditionally concatenate it. Otherwise the agent is told the
    correct answer up-front and asked to demonstrate the analytic process
    that justifies it — the produced trajectory becomes a reference for
    future similar tasks at retrieval time. The agent is still required
    to emit its final answer in the same format the base prompt requests
    so downstream parsers and metrics work unchanged.
    """
    if ground_truth is None:
        return ""
    if isinstance(ground_truth, (dict, list)):
        gt_str = json.dumps(ground_truth, ensure_ascii=False, default=str)
    else:
        gt_str = str(ground_truth)
    return (
        "\n\n---\n"
        "TRAINING MODE — the correct answer to the above is:\n"
        f"  {gt_str}\n\n"
        "Do TWO things before emitting your final answer:\n\n"
        "1. Inspect the series with whatever tools are available and walk through\n"
        "   the reasoning step by step.\n\n"
        "2. Output a <context_to_forecast>...</context_to_forecast> block\n"
        "   (≤ 3 sentences) that explicitly states:\n"
        "     - which sentences in the context (background / scenario / constraints\n"
        "       / question / options) drive the answer's shape, AND\n"
        "     - the rule that maps those sentences to that shape (e.g. \"scenario\n"
        "       says 'heat wave for 2 hours' → ground truth has a 4x spike at the\n"
        "       stated start → because air-conditioning load scales with cooling\n"
        "       demand\").\n"
        "   Make this block transferable, NOT series-specific — future test-time\n"
        "   agents on similar tasks will read it to map their own context to a\n"
        "   forecast shape, so avoid statements like \"the mean is 850\" and prefer\n"
        "   statements like \"scenario X implies pattern Y because of mechanism Z\".\n\n"
        "Then produce your final answer in the format requested above."
    )


# ---------------------------------------------------------------------------
# TSRBench
# ---------------------------------------------------------------------------

_TSRBENCH_TOOLS_HINT = (
    "The numerical time series for this question is already loaded into your "
    "analysis tool server. Inspect it with the available tools before "
    "answering. A typical pattern:\n"
    "  1. list_channels() or series_overview() to see what channels exist\n"
    "  2. channel_stats / channel_values / compute_acf / detect_periodicity / "
    "find_peaks on individual channels to gather the evidence you need\n"
    "Do NOT guess values you haven't retrieved through tools."
)


def build_tsrbench_prompt_with_tools(
    record: dict,
    *,
    ground_truth: Any = None,
    memory_references: list[str] | None = None,
) -> str:
    """Build a TSRBench prompt that points the agent at its MCP tool server.

    The numeric series is preloaded into the agent's MCP server by the
    harness, so the prompt omits any inline rendering of it. Pass
    ``ground_truth`` to switch into training mode and ``memory_references``
    at test time to inject a retrieval block.
    """
    question = record.get("question", "").strip()
    names = record.get("name_of_series")
    domain = record.get("domain", "")
    choices = record.get("choices")
    task = record.get("task", "")
    prefix = _retrieval_prefix(memory_references)
    suffix = _training_suffix(ground_truth)

    if isinstance(names, str):
        try:
            names = json.loads(names)
        except Exception:
            pass

    header_parts = []
    if domain:
        header_parts.append(f"Domain: {domain}")
    if task:
        header_parts.append(f"Task type: {task}")
    if names:
        header_parts.append(f"Series names: {names}")
    header = "\n".join(header_parts)

    # Perception's `<ts><ts/>` placeholder: replace with a tools-pointing stub
    # rather than rendered values.
    if "<ts><ts/>" in question:
        filled = question.replace(
            "<ts><ts/>",
            "[time series loaded — use the analysis tools to inspect it]",
        )
        head = f"{header}\n\n" if header else ""
        return (
            f"{prefix}{head}{_TSRBENCH_TOOLS_HINT}\n\n"
            f"{filled}\n\n"
            f"After your reasoning, output your final answer wrapped exactly like "
            f"this on its own line: <answer>X</answer> where X is a single letter."
            f"{suffix}"
        )

    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except Exception:
            choices = [choices]
    if isinstance(choices, dict):
        choices = [choices[k] for k in sorted(choices.keys())]

    if isinstance(choices, list) and choices:
        options_text = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(choices)
        )
        return (
            f"{prefix}{header}\n\n"
            f"{_TSRBENCH_TOOLS_HINT}\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Reason carefully, then output your final answer as a single letter "
            f"wrapped exactly like this on its own line: <answer>X</answer>"
            f"{suffix}"
        )

    return (
        f"{prefix}{header}\n\n"
        f"{_TSRBENCH_TOOLS_HINT}\n\n"
        f"Question:\n{question}\n\n"
        f"Reason carefully, then output your final answer wrapped exactly like: "
        f"<answer>YOUR_ANSWER</answer>. If the answer is numeric, the wrapped "
        f"value must be a single number."
        f"{suffix}"
    )


# ---------------------------------------------------------------------------
# TSAIA
# ---------------------------------------------------------------------------

def build_tsaia_analysis_prompt(
    record: dict,
    *,
    ground_truth: Any = None,
    memory_references: list[str] | None = None,
) -> str:
    """Build a prompt for a TSAIA analysis_questions row.

    ``ground_truth`` switches the prompt into training mode (see
    ``_training_suffix``). For analysis rows the GT is typically the
    ``ground_truth_data`` dict / scalar / list — passed through verbatim.
    """
    prompt = record.get("prompt", "").strip()
    data_str = record.get("data_str", "")
    context = record.get("context", "")
    constraint = record.get("constraint", "")
    executor_variables = record.get("executor_variables", "")

    parts = [prompt]
    if context:
        parts.append(f"\nContext:\n{context}")
    if executor_variables:
        parts.append(f"\nAvailable variables / data:\n{executor_variables}")
    if data_str:
        parts.append(f"\nData:\n{data_str}")
    if constraint:
        parts.append(f"\nOutput constraint:\n{constraint}")
    parts.append(
        "\n\nReason carefully, then output your final answer wrapped exactly like: "
        "<answer>YOUR_ANSWER</answer>. If the answer is a number, wrap a single "
        "number. If the answer is a JSON value, wrap a single valid JSON literal."
    )
    return _retrieval_prefix(memory_references) + "\n".join(parts) + _training_suffix(ground_truth)


_TSAIA_MC_TOOLS_HINT = (
    "The numeric data for this MC question has been loaded into your analysis\n"
    "tool server. DO NOT try to read prices out of the prompt text — the\n"
    "prompt has at most a truncated inline copy. Use the tools:\n"
    "  1. list_channels() / series_overview() to see what is loaded.\n"
    "  2. For VaR / SharpeRatio (portfolio questions), channels follow the\n"
    "     naming convention ``A.stock_0``, ``A.stock_1``, ..., ``B.stock_0``,\n"
    "     ..., ``C.stock_N``. SharpeRatio also has ``A.riskfree``,\n"
    "     ``B.riskfree``, ``C.riskfree``. series_overview()['meta']\n"
    "     ['option_weights'] gives the per-option weight vector and ['horizon']\n"
    "     gives the forecast horizon in periods (already parsed from the prompt).\n"
    "  3. For MarketAB-alpha / MarketAB-beta, channels are ``asset`` and\n"
    "     ``market``.\n"
    "  4. Compute the quantity directly with one of the quant tools:\n"
    "       - portfolio_var(channels, weights, horizon, alpha, method)\n"
    "       - portfolio_sharpe(channels, weights, risk_free, period_per_year)\n"
    "       - capm_regression(asset_channel, market_channel)\n"
    "     ARIMA forecasting is available via arima_forecast(name, periods,\n"
    "     p, d, q) when the question asks about future values.\n"
    "Do NOT guess from the prompt; ground each comparison in a tool output."
)


def build_tsaia_mc_prompt_with_tools(
    record: dict,
    *,
    ground_truth: Any = None,
    memory_references: list[str] | None = None,
) -> str:
    """Build a TSAIA MC prompt that points the agent at its MCP tool server.

    The canonical numeric data is preloaded into the agent's MCP server by
    the runner (more reliable than the truncated inline copy in the prompt
    text).
    """
    prompt = record.get("prompt", "").strip()
    options = record.get("options")
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            options = [options]
    options_text = ""
    if isinstance(options, list):
        options_text = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )
    answer_info = record.get("answer_info", "")

    parts = [_TSAIA_MC_TOOLS_HINT, "", prompt]
    if options_text:
        parts.append(f"\nOptions:\n{options_text}")
    if answer_info:
        parts.append(f"\nAnswer format:\n{answer_info}")
    parts.append(
        "\n\nReason carefully (cite the tool outputs you base each comparison on),\n"
        "then output your final answer as a single letter wrapped exactly like\n"
        "this on its own line: <answer>X</answer>"
    )
    return _retrieval_prefix(memory_references) + "\n".join(parts) + _training_suffix(ground_truth)


# ---------------------------------------------------------------------------
# CiK
# ---------------------------------------------------------------------------

_CIK_TOOLS_HINT = (
    "The past_time history is already loaded into your analysis tool server "
    "as one channel per series. Inspect it with the available tools before "
    "writing your forecast — a typical pattern:\n"
    "  1. series_overview() to see channel names and counts\n"
    "  2. channel_stats / compute_acf / detect_periodicity / find_peaks to\n"
    "     measure level, dispersion, seasonality, and trend\n"
    "  3. channel_values(name, start, end, stride) to read raw segments\n"
    "Do NOT just eyeball the inline JSON; ground every claim about the past\n"
    "in a tool output."
)


def build_cik_prompt(
    record: dict,
    *,
    want_quantiles: bool = True,
    ground_truth: Any = None,
    memory_references: list[str] | None = None,
    with_tools_hint: bool = True,
) -> str:
    """Build a prompt for one CiK record.

    The agent is asked to emit a JSON forecast. If ``want_quantiles`` is True
    we ask for a 9-quantile forecast; otherwise just a point forecast.
    ``past_time`` is always inlined as JSON (CiK histories are short — a few
    dozen points typically). When ``with_tools_hint`` (default), the prompt
    also tells the agent to inspect past_time via MCP tools rather than
    reading the JSON directly; this is what produces rich trajectories for
    the memory bank.
    """
    background = record.get("background", "") or ""
    scenario = record.get("scenario", "") or ""
    constraints = record.get("constraints", "") or ""

    past_time = record.get("past_time")
    future_time = record.get("future_time")
    if isinstance(past_time, str):
        try:
            past_time = json.loads(past_time)
        except Exception:
            past_time = {}
    if isinstance(future_time, str):
        try:
            future_time = json.loads(future_time)
        except Exception:
            future_time = {}

    horizon = 0
    n_channels = 0
    if isinstance(future_time, dict) and future_time:
        n_channels = len(future_time)
        first_col = next(iter(future_time.values()))
        if isinstance(first_col, dict):
            horizon = len(first_col)
    total_length = horizon * n_channels

    if want_quantiles:
        out_spec = (
            "Output your forecast as a single fenced JSON block of the form:\n"
            "```json\n"
            '{"levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],\n'
            f' "samples": [[q1_t1,...,q9_t1], ...]}}  '
            f"// samples must be a list of EXACTLY {total_length} elements"
            f" ({n_channels} channel(s) × {horizon} step(s), concatenated in channel order),"
            " each a list of 9 values at the levels above\n"
            "```\n"
            "If you can only produce a point forecast, instead output:\n"
            "```json\n"
            f'{{"forecast": [v1, v2, ..., v{total_length}]}}\n'
            "```"
        )
    else:
        out_spec = (
            "Output your forecast as a single fenced JSON block of the form:\n"
            "```json\n"
            f'{{"forecast": [v1, v2, ..., v{total_length}]}}\n'
            "```"
        )

    parts = []
    if with_tools_hint:
        parts.append(_CIK_TOOLS_HINT)
    if background.strip():
        parts.append(f"Background:\n{background.strip()}")
    if scenario.strip():
        parts.append(f"Scenario:\n{scenario.strip()}")
    if constraints.strip():
        parts.append(f"Constraints:\n{constraints.strip()}")

    parts.append(
        f"Historical data (timestamp -> value, by channel):\n{json.dumps(past_time)}"
    )
    parts.append(
        f"You must forecast the next {horizon} step(s) for EACH of the "
        f"{n_channels} channel(s), then concatenate them in the same channel "
        f"order as the historical data, producing a TOTAL of EXACTLY "
        f"{total_length} numeric values. The harness validates length strictly: "
        f"any other count is treated as a parse failure."
    )
    parts.append(out_spec)
    return _retrieval_prefix(memory_references) + "\n\n".join(parts) + _training_suffix(ground_truth)
