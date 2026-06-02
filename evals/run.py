"""Run the evaluation: Ouroboros vs. single-shot baseline, judged pairwise.

Usage:
    python -m evals run [--limit N]
    python -m evals plot

Requires a generation API key (e.g. free Groq tier) in .env. Results and an
LLM response cache are written under EVAL_OUT_DIR (default evals/results/), so
re-runs are cheap and reproducible.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evals.config import load_eval_config
from evals.engine import run_baseline, run_ouroboros
from evals.judge import compare
from evals.tasks import get_tasks
from ouroboros.config import get_settings
from ouroboros.providers import get_llm
from ouroboros.usage import configure_tracing

load_dotenv()


def _enable_cache(out_dir: Path) -> None:
    """Cache LLM responses to disk for reproducibility + cheap re-runs."""
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        set_llm_cache(SQLiteCache(str(out_dir / "llm_cache.db")))
    except Exception:
        pass  # caching is an optimization, not a requirement


def _resolved_models(cfg) -> tuple[str, str]:
    settings = get_settings()
    default = {
        "groq": settings.groq_model,
        "openai": settings.openai_model,
        "ollama": settings.ollama_model,
    }.get(settings.llm_provider, settings.groq_model)
    return cfg.gen_model or default, cfg.judge_model or default


def _summarize(results: list[dict]) -> dict:
    overall = Counter(r["winner"] for r in results)
    by_mode: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_mode[r["mode"]][r["winner"]] += 1
    n = len(results) or 1
    return {
        "total": len(results),
        "ouroboros_wins": overall["ouroboros"],
        "baseline_wins": overall["baseline"],
        "ties": overall["tie"],
        "ouroboros_win_rate": round(overall["ouroboros"] / n, 3),
        "baseline_win_rate": round(overall["baseline"] / n, 3),
        "by_mode": {
            mode: dict(counts) for mode, counts in sorted(by_mode.items())
        },
    }


async def run_eval(limit: int | None = None) -> dict:
    cfg = load_eval_config()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _enable_cache(out_dir)
    configure_tracing()

    gen_model, judge_model = _resolved_models(cfg)
    if gen_model == judge_model:
        print(
            f"⚠  Judge and generator are the same model ({gen_model}). For less biased "
            "results set EVAL_JUDGE_MODEL to a different free model.\n"
        )

    # Generator (raw model — Ouroboros needs bind_tools and has its own fallbacks).
    gen_llm = get_llm(temperature=cfg.gen_temperature, model=cfg.gen_model)
    judge_llm = get_llm(temperature=0.0, model=cfg.judge_model)

    tasks = get_tasks(limit)
    results: list[dict] = []
    for t in tasks:
        ouro = await run_ouroboros(gen_llm, t.seed, t.mode, cfg.max_cycles)
        base = await run_baseline(gen_llm, t.seed, t.mode)
        winner = await compare(judge_llm, t.seed, ouro, base)
        results.append(
            {
                "id": t.id,
                "mode": t.mode,
                "seed": t.seed,
                "ouroboros": ouro,
                "baseline": base,
                "winner": winner,
            }
        )
        print(f"[{t.id}] {t.mode:<13} winner = {winner}")

    summary = _summarize(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gen_model": gen_model,
            "judge_model": judge_model,
            "gen_temperature": cfg.gen_temperature,
            "max_cycles": cfg.max_cycles,
        },
        "summary": summary,
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(summary, out_dir)
    return payload


def _print_summary(summary: dict, out_dir: Path) -> None:
    print("\n" + "=" * 48)
    print("Ouroboros vs. single-shot baseline")
    print("=" * 48)
    print(f"  Tasks judged:    {summary['total']}")
    print(f"  Ouroboros wins:  {summary['ouroboros_wins']}  ({summary['ouroboros_win_rate']:.0%})")
    print(f"  Baseline wins:   {summary['baseline_wins']}  ({summary['baseline_win_rate']:.0%})")
    print(f"  Ties:            {summary['ties']}")
    print(f"\n  Results: {out_dir / 'results.json'}")
    print("  Make the chart:  python -m evals plot")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="evals", description="Ouroboros evaluation harness")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run the evaluation")
    run_p.add_argument("--limit", type=int, default=None, help="Only run the first N tasks")
    sub.add_parser("plot", help="Render the chart from results.json")

    args = parser.parse_args(argv)
    if args.command == "run":
        asyncio.run(run_eval(args.limit))
    elif args.command == "plot":
        from evals.plot import plot_results

        plot_results()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
