"""Metacognitive controller: content-aware halting for adaptive test-time compute.

This is the research wedge (see docs/RESEARCH.md). Standard reflection loops spend
a *fixed* amount of compute — N iterations, a depth cap, or (as in this engine's
legacy router) a coin flip. The controller replaces that with a principled stop
decision driven by cheap internal signals:

- **answer stability** — semantic convergence between successive refined answers
  (cosine over the same embedder used for semantic memory). High stability means
  the loop has stopped changing its mind: more cycles buy little.
- **self-confidence** — the synthesizer's own 0-1 estimate of how settled its
  answer is.

Bounded by a compute budget, this spends more cycles on hard / still-moving
problems and halts early on ones that have already converged. Because halting
early stops wasting tokens, the same mechanism is what keeps the engine inside
free-tier rate limits — the product virtue and the research claim are one thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ouroboros.memory import cosine_similarity, get_embedder
from ouroboros.models import OuroborosConfig


def answer_stability(prev: str, curr: str) -> float:
    """Cosine similarity between two successive answers (1.0 = same meaning).

    Uses the shared embedder, so it works offline via the lexical fallback when
    sentence-transformers is not installed. Returns 0.0 if either side is empty
    (i.e. the first cycle, where there is nothing to compare against yet).
    """
    if not prev or not curr:
        return 0.0
    a, b = get_embedder().embed([prev, curr])
    return max(0.0, min(1.0, cosine_similarity(a, b)))


@dataclass(frozen=True)
class Decision:
    """A halting decision plus the reason, for routing and interpretability."""

    halt: bool
    reason: str


def decide(
    *, depth: int, stability: float, confidence: float, config: OuroborosConfig
) -> Decision:
    """Decide whether to stop refining, from internal signals + the budget.

    Pure function of scalars (no LLM, no embedding) so it is fully unit-testable.
    Precedence: honour ``min_cycles`` first, then the hard ``compute_budget`` cap,
    then convergence (stable *and* confident), then diminishing returns (stable
    but not yet confident — the answer has stopped moving regardless).
    """
    if depth < config.min_cycles:
        return Decision(False, "min_cycles")
    if depth >= config.compute_budget:
        return Decision(True, "budget")
    if (
        stability >= config.stability_threshold
        and confidence >= config.confidence_threshold
    ):
        return Decision(True, "converged")
    if stability >= config.stability_threshold:
        return Decision(True, "no_marginal_gain")
    return Decision(False, "continue")
