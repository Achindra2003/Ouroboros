"""Evaluation configuration (env-driven, kept separate from app settings).

Key knob for credibility: the judge should be a *different* model than the
generator, so the judge does not simply prefer its own style. Override via env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    # Generator: used for BOTH Ouroboros and the single-shot baseline (fair comparison).
    gen_model: str | None
    # Judge: a different free model, scored at temperature 0 for determinism.
    judge_model: str | None
    gen_temperature: float
    max_cycles: int
    out_dir: str


def load_eval_config() -> EvalConfig:
    return EvalConfig(
        gen_model=os.environ.get("EVAL_GEN_MODEL") or None,
        judge_model=os.environ.get("EVAL_JUDGE_MODEL") or None,
        gen_temperature=float(os.environ.get("EVAL_GEN_TEMPERATURE", "0.7")),
        max_cycles=int(os.environ.get("EVAL_MAX_CYCLES", "6")),
        out_dir=os.environ.get("EVAL_OUT_DIR", "evals/results"),
    )
