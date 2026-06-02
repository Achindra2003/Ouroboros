"""Render the eval results as a chart (overall + per-mode win rates)."""

from __future__ import annotations

import json
from pathlib import Path

from evals.config import load_eval_config


def plot_results(results_path: str | None = None, out_path: str | None = None) -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless / CI-safe
    import matplotlib.pyplot as plt

    cfg = load_eval_config()
    out_dir = Path(cfg.out_dir)
    results_file = Path(results_path) if results_path else out_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"{results_file} not found. Run `python -m evals run` first."
        )

    payload = json.loads(results_file.read_text(encoding="utf-8"))
    summary = payload["summary"]
    chart_path = Path(out_path) if out_path else out_dir / "chart.png"

    modes = sorted(summary["by_mode"])
    ouro = [summary["by_mode"][m].get("ouroboros", 0) for m in modes]
    base = [summary["by_mode"][m].get("baseline", 0) for m in modes]
    ties = [summary["by_mode"][m].get("tie", 0) for m in modes]

    fig, (ax_overall, ax_mode) = plt.subplots(1, 2, figsize=(12, 5))

    # Overall stacked-ish comparison.
    labels = ["Ouroboros", "Baseline", "Tie"]
    counts = [summary["ouroboros_wins"], summary["baseline_wins"], summary["ties"]]
    colors = ["#e8c060", "#888888", "#cccccc"]
    ax_overall.bar(labels, counts, color=colors)
    ax_overall.set_title(
        f"Overall wins (n={summary['total']})\n"
        f"Ouroboros {summary['ouroboros_win_rate']:.0%} vs Baseline {summary['baseline_win_rate']:.0%}"
    )
    ax_overall.set_ylabel("tasks won")
    for i, c in enumerate(counts):
        ax_overall.text(i, c, str(c), ha="center", va="bottom")

    # Per-mode grouped bars.
    x = range(len(modes))
    w = 0.27
    ax_mode.bar([i - w for i in x], ouro, width=w, label="Ouroboros", color="#e8c060")
    ax_mode.bar(list(x), base, width=w, label="Baseline", color="#888888")
    ax_mode.bar([i + w for i in x], ties, width=w, label="Tie", color="#cccccc")
    ax_mode.set_xticks(list(x))
    ax_mode.set_xticklabels(modes, rotation=20)
    ax_mode.set_title("Wins by mode")
    ax_mode.set_ylabel("tasks")
    ax_mode.legend()

    model_info = payload.get("config", {})
    fig.suptitle(
        f"Recursive introspection vs. single-shot  ·  gen={model_info.get('gen_model')}  "
        f"judge={model_info.get('judge_model')}",
        fontsize=10,
    )
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=130)
    print(f"Chart written to {chart_path}")
    return str(chart_path)


if __name__ == "__main__":
    plot_results()
