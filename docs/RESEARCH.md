# Ouroboros — Research Direction (Wedge A: Metacognitive Adaptive Compute)

This is the working design doc for turning Ouroboros from a portfolio piece into
**(a) a product that runs on free tier** and **(b) the basis of a credible research
paper**. It records the thesis, what's already built, and what's deferred (and why),
so the next session can resume without re-deriving context.

## 1. The thesis (one sentence)

> A **metacognitive controller** that halts iterative reasoning based on cheap
> internal signals — answer **stability** (semantic convergence) and self-reported
> **confidence** — matches fixed-depth reasoning quality while spending materially
> fewer tokens, with the savings growing as problem difficulty varies.

Why this is a defensible wedge (and "just reflect more" is not):

- The reflection space is saturated (Self-Refine, Reflexion, Tree-of-Thoughts,
  multi-agent debate). Novelty must be specific.
- Known landmine: *"LLMs Cannot Self-Correct Reasoning Yet"* (Huang et al.) — pure
  self-reflection **without external feedback often makes models worse**. Our own
  first eval reproduced a version of this (see §2).
- The unsaturated, currently-hot frontier is **test-time compute allocation**
  (o1-style reasoning). *Adaptive, content-aware* allocation — deciding *how much*
  to think per problem — is comparatively underexplored. That's the gap.
- Product virtue and research claim are the **same mechanism**: halting early stops
  wasting tokens → lower cost → and it's literally what keeps us under Groq's
  100K-tokens/day free-tier cap.

## 2. What motivated this: the first eval (negative result)

First real run (`evals/results/results.json`, gen `llama-3.1-8b-instant`, judge
`openai/gpt-oss-20b`, 3 cycles, 15 tasks, blind position-swapped judge):

**Ouroboros 0 wins · baseline 14 · 1 tie.** Root causes, from reading all 15 pairs:

1. **Output-shape mismatch (harness):** the engine returned a one-line *surfaced
   insight* (~35 words) vs. the baseline's full answer (~100 words). Haiku vs essay.
2. **Divergence / drift:** no anchor, no critic → each cycle wandered off the seed
   (e.g. "memory as a choice" → "I'm a narrative construct in a larger story").
3. **Leaked planning fragments:** solve-mode answers literally began "The next step
   is to…" — internal meta-thought escaping as the final answer.

Diagnosis: the loop did **divergent** recursion (generate new tangents) when what
wins is **convergent** recursion (draft → critique → improve = Self-Refine).

## 3. What is BUILT (Phase 1 — product, ships on free tier)

Behind the `config.adaptive` flag (default **off**, so the legacy engine and all
existing tests are unchanged). 124 tests pass, ruff clean.

- **`ouroboros/graph/controller.py`** — the contribution:
  - `answer_stability(prev, curr)` — cosine over the shared embedder (offline
    lexical fallback), measuring semantic convergence between successive answers.
  - `decide(depth, stability, confidence, config)` — pure, unit-tested halting
    decision. Precedence: `min_cycles` → `compute_budget` cap → converged
    (stable ∧ confident) → diminishing returns (stable) → continue. Returns a
    `Decision(halt, reason)` for routing **and** interpretability.
- **Convergent self-refine loop** — `_synthesize_adaptive` in `nodes.py`:
  anchored to the seed (fixes drift), refines the *current best answer*
  (fixes divergence), emits a `CONFIDENCE: <0-1>` marker parsed by
  `_parse_confidence`, computes stability, and stores the controller decision.
- **Output parity** — `make_surface` in adaptive mode passes the converged
  synthesis through as the answer (no aphorism, no leaked fragment).
- **Routing** — `route_after_synthesis` honours the controller's `should_halt`;
  `route_after_breathe` ends after one converged answer per run.
- **CLI** — `ouroboros run "..." --adaptive [--max-cycles N]` (N → `compute_budget`).
- **Config** (`OuroborosConfig`): `adaptive`, `compute_budget`, `min_cycles`,
  `stability_threshold`, `confidence_threshold`.

Smoke-verified end-to-end: on a stable answer the controller halts at 2 cycles
("converged") instead of running the full budget — adaptive savings demonstrated.

## 4. What is DEFERRED (needs time and/or a few dollars)

### Phase 1.5 — the product PROOF (free tier, cheap, do next)
Show **equal-or-better pairwise quality at fewer tokens**. Concretely:
- Add a token meter to `evals/engine.py` (attach a `new_usage_handler()` per run;
  record `total_tokens` alongside the verdict).
- Run three arms on the existing 15 tasks (8B): **single-shot**, **fixed-depth-6**,
  **adaptive**. Report a small table: win-rate (blind judge) **and** mean tokens/task.
- Target result: adaptive ≈ fixed-depth quality at materially fewer tokens, and
  ≥ single-shot quality. That's the shippable claim + the README chart.
- This also fixes the original 0/15 honestly: re-run with the convergent loop and
  report what actually happens (including modes where it still ties/loses).

### Phase 2 — product surface (visualization = interpretability)
Expose the controller in the live web UI: a "thinking budget" meter that drains as
cycles spend, the per-cycle **stability** curve climbing toward the halt threshold,
and the **stop reason** ("converged" / "budget" / "no marginal gain"). This is
simultaneously the layman "watch it decide how hard to think" hook and the paper's
interpretability story. Keep the existing mood/energy/perspective machinery as the
*substrate the controller observes* (perspectives = diverse critics; energy =
interpretable budget) — keeps the portfolio appeal, gains the rigorous spine.

### Phase 3 — the PAPER rig (needs investment: API budget + runtime)
- **Benchmarks with checkable answers** (so we measure *accuracy*, not vibes):
  GSM8K / MATH (reasoning), plus a hard open-ended QA set with a graded rubric.
  Add adapters under `evals/` that map a benchmark item → seed + scorer.
- **Strong baselines at matched compute** (the bar for acceptance): single-shot,
  best-of-N, **Self-Refine** (fixed N), **Reflexion**. Adaptive must win on the
  **accuracy-per-token** curve, not just beat single-shot.
- **Stronger generator** than free-tier 8B (the 8B ceiling limits how convincing
  the numbers are) + **multiple seeds** for error bars / significance.
- **Ablations:** stability-only vs confidence-only vs both; sensitivity to
  `stability_threshold`; learned vs heuristic controller; does adaptive's token
  saving grow with difficulty variance (the headline claim)?
- **Headline figure:** accuracy-per-token Pareto curve, adaptive dominating the
  fixed-depth family.

### Later / optional — second wedge (fusion)
Cross-episode **experiential memory**: does remembering its own past insights across
a *sequence* of related problems produce measurable transfer (continual in-context
self-improvement, no weight updates)? The semantic memory already accumulates across
sessions; this is a natural Phase-4 paper extension and a strong product story
("gets sharper on your domain the more you use it").

## 5. The free-tier ↔ paper split (the deal)

| | Free tier (product) | Paper (investment) |
|---|---|---|
| Model | Groq free 8B | stronger generator |
| Evidence | pairwise quality **at fewer tokens** | accuracy-per-token vs strong baselines |
| Tasks | the 15-task set | GSM8K / MATH / graded QA |
| Runs | single | multi-seed, error bars |

Same code, two evidence tiers: the product ships on free tier; the paper is the same
system pointed at harder benchmarks with a bigger budget.
