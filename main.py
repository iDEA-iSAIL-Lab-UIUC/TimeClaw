"""CLI entry point for running TimeClaw benchmark evaluations.

Examples:
    python main.py --benchmark cik       --model gpt-5-nano --num-workers 16
    python main.py --benchmark tsrbench  --model gpt-5-nano --ratio 0.2 --subset all
    python main.py --benchmark tsaia     --model gpt-5-nano --ratio 1.0 --subset mc

Output: ./results/{benchmark}/{model}_{mode}[_{subset}][_k{k}]_{date}-{time}/
        {predictions.jsonl, trajectories.jsonl, summary.json, run_config.json}
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

from timeclaw.agents import TimeClaw


def _setup_env() -> None:
    """Load API keys from .env if present.

    ``override=True`` so that .env wins over the shell environment. Some
    shells export keys as empty strings, which would otherwise silently
    shadow the real value in .env.
    """
    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "[warn] OPENAI_API_KEY not set in environment. Put it in ./.env or "
            "export it before running OpenAI models.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="TimeClaw benchmark evaluation harness")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["cik", "tsrbench", "tsaia"],
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="Model id (OpenAI gpt-*, Gemini gemini-*, or Anthropic claude-*).",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=4,
        help="Max concurrent agent invocations (== MCP slot pool size).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Within each task family, keep ceil(ratio * group_size) samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic subsampling.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help=(
            "Benchmark-specific subset filter. "
            "TSRBench: one of [all, perception, prediction, decision, reasoning, "
            "or a specific subtask like causal_reasoning]. "
            "TSAIA: one of [all, analysis, mc, anomaly, causal, forecasting, "
            "decision, risk_metric, or a specific question_type]. "
            "CiK: ignored."
        ),
    )
    parser.add_argument(
        "--no-quantiles",
        dest="want_quantiles",
        action="store_false",
        help="For CiK, ask the agent for point forecast only (skip quantile JSON).",
    )
    parser.set_defaults(want_quantiles=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "test"],
        help=(
            "Train/test split mode. 'full' evaluates on the entire subsampled "
            "set. 'train' evaluates on the train half and writes trajectories "
            "to the memory bank (ground truth is revealed to the agent). "
            "'test' evaluates on the test half and injects top-k retrieved "
            "memories. The split is taken AFTER --ratio subsampling, within "
            "each family."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of each family that goes to train when --mode is 'train' "
            "or 'test'. Every family is guaranteed >=1 train record so "
            "retrieval at test time always has a same-family exemplar. "
            "Ignored when --mode=full."
        ),
    )
    parser.add_argument(
        "--split-seed",
        dest="split_seed",
        type=int,
        default=2026,
        help="Seed for the within-family train/test split. Independent of --seed.",
    )
    parser.add_argument(
        "--k-neighbors",
        dest="k_neighbors",
        type=int,
        default=3,
        help=(
            "Number of top retrieved memories to inject into each test prompt "
            "when --mode=test. Ignored in full / train mode."
        ),
    )
    parser.add_argument(
        "--retrieve-same-family-only",
        dest="retrieve_same_family_only",
        action="store_true",
        help=(
            "Diagnostic switch: at test time, restrict retrieval to records "
            "with the same family_key as the query (TSRBench: _subtask, TSAIA: "
            "_question_type, CiK: name). Off by default."
        ),
    )
    parser.add_argument(
        "--text-filter-size",
        dest="text_filter_size",
        type=int,
        default=0,
        help=(
            "Two-stage retrieval: first filter to top-N candidates by "
            "text-embedding-3-small cosine on the task's NL context, then "
            "rank top-k by fingerprint L2 within those. 0 disables the text "
            "filter (pure fingerprint, default). 20 is a sensible value when "
            "enabled. Training-mode runs always embed and store text when "
            "this is >0 so the bank is ready for test-time retrieval."
        ),
    )
    parser.add_argument(
        "--cik-samples",
        dest="cik_samples",
        type=int,
        default=None,
        help=(
            "[CiK only] Override --ratio: take the first N records of each "
            "family by seed order (5 seeds per family in CiK). When set, "
            "--ratio is ignored for CiK."
        ),
    )
    parser.add_argument(
        "--cik-train-samples",
        dest="cik_train_samples",
        type=int,
        default=None,
        help=(
            "[CiK only] Override --train-ratio: of the per-family subsample, "
            "the first N (by seed order) go to train, the rest to test. "
            "Requires --cik-samples to also be set."
        ),
    )
    parser.add_argument(
        "--cik-context",
        dest="cik_context",
        type=str,
        default="all",
        help=(
            "[CiK only] Filter task families by context type. One of "
            "[all, history, future, intemporal, covariates, causal] or a "
            "comma-separated combination (UNION; e.g. 'causal,future'). A task "
            "family can belong to multiple context types. The filter is "
            "applied BEFORE --ratio / --cik-samples subsampling."
        ),
    )

    args = parser.parse_args()
    _setup_env()

    agent = TimeClaw(model=args.model, pool_size=args.num_workers)

    async def _go() -> str:
        try:
            if args.benchmark == "cik":
                from timeclaw.evaluation.cik import run_cik

                return await run_cik(
                    agent,
                    mode=args.mode,
                    train_ratio=args.train_ratio,
                    split_seed=args.split_seed,
                    k_neighbors=args.k_neighbors,
                    retrieve_same_family_only=args.retrieve_same_family_only,
                    text_filter_size=args.text_filter_size,
                    cik_samples=args.cik_samples,
                    cik_train_samples=args.cik_train_samples,
                    cik_context=args.cik_context,
                    num_workers=args.num_workers,
                    ratio=args.ratio,
                    seed=args.seed,
                    want_quantiles=args.want_quantiles,
                )
            if args.benchmark == "tsrbench":
                from timeclaw.evaluation.tsrbench import run_tsrbench

                return await run_tsrbench(
                    agent,
                    mode=args.mode,
                    train_ratio=args.train_ratio,
                    split_seed=args.split_seed,
                    k_neighbors=args.k_neighbors,
                    retrieve_same_family_only=args.retrieve_same_family_only,
                    text_filter_size=args.text_filter_size,
                    num_workers=args.num_workers,
                    ratio=args.ratio,
                    seed=args.seed,
                    subset=args.subset,
                )
            if args.benchmark == "tsaia":
                from timeclaw.evaluation.tsaia import run_tsaia

                return await run_tsaia(
                    agent,
                    mode=args.mode,
                    train_ratio=args.train_ratio,
                    split_seed=args.split_seed,
                    k_neighbors=args.k_neighbors,
                    retrieve_same_family_only=args.retrieve_same_family_only,
                    text_filter_size=args.text_filter_size,
                    num_workers=args.num_workers,
                    ratio=args.ratio,
                    seed=args.seed,
                    subset=args.subset,
                )
            raise ValueError(f"Unknown benchmark: {args.benchmark}")
        finally:
            await agent.close()

    run_dir = asyncio.run(_go())
    print(f"\nDone. Results in: {run_dir}")


if __name__ == "__main__":
    main()
