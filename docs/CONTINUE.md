# Continuing Ouroboros on a new machine

This repo is self-contained: a fresh `git clone` has everything you need. The only
thing that does **not** travel with the repo is your Groq API key (it lives in a
gitignored `.env`). This doc + [`RESEARCH.md`](RESEARCH.md) are the source of truth
for where the project is and what's next.

## 1. One-time setup on a fresh machine

```bash
git clone https://github.com/Achindra2003/Ouroboros.git
cd Ouroboros

# Python 3.11+ required. Create a venv, then install with the extras you want:
pip install -e ".[dev,memory,eval]"
#   dev    → pytest + ruff
#   memory → real semantic embeddings (sentence-transformers)
#   eval   → evaluation harness charting (matplotlib)
#   also available: local (Ollama), persist (sqlite checkpointing), search (Tavily)
```

### Bring your secret (not in git)
Create `.env` in the repo root (copy `.env.example` and fill in the key):

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...        # your free key from https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile

# For the eval harness (8B generator, different-family judge for anti-bias):
EVAL_GEN_MODEL=llama-3.1-8b-instant
EVAL_JUDGE_MODEL=openai/gpt-oss-20b
EVAL_MAX_CYCLES=3
```

> Free-tier note: `llama-3.3-70b-versatile` has only **100K tokens/day** — one full
> introspection run nearly exhausts it. Eval/experiment on `llama-3.1-8b-instant`
> (much higher limit). This is exactly why the adaptive controller matters.

## 2. Verify the clone is healthy

```bash
pytest -q                 # expect 124 passed
ruff check ouroboros/     # expect: All checks passed!
```

## 3. Run it

```bash
# Legacy engine
ouroboros run "What is consciousness?"

# Adaptive controller (the current work — convergent self-refine + content-aware halting)
ouroboros run "What is the link between attention and identity?" --adaptive --no-steer

# Web UI (live "watch it think")
uvicorn ouroboros.server:app --reload     # http://localhost:8000

# Eval harness
python -m evals run        # → evals/results/results.json
python -m evals plot       # → evals/results/chart.png
```

## 4. Where the project is right now (2026-06-07)

**Goal (shifted):** beyond a portfolio piece — make it (a) a product that runs on
free tier and (b) the basis of a credible research paper. The wedge is **Wedge A:
metacognitive adaptive test-time-compute allocation** (see `RESEARCH.md` §1).

**Built (Phase 1, this branch/commit):** the adaptive controller behind a
`config.adaptive` flag — convergent self-refine + content-aware halting
(`ouroboros/graph/controller.py`). 124 tests pass, ruff clean. Smoke-verified it
halts early on convergence. This also fixes the failures behind the first eval.

**Motivating result:** the *legacy* divergent loop lost **0/15** to single-shot
(`evals/results/results.json`) — drift, output-shape mismatch, leaked planning
fragments. The convergent loop was built to fix all three. See `RESEARCH.md` §2–3.

## 5. The runway (do these in order; all free tier until the wall)

1. **Phase 1.5 — the product proof (DO THIS NEXT):** add a token meter to
   `evals/engine.py`; run 3 arms (single-shot / fixed-depth-6 / adaptive) on the 15
   tasks; produce a **win-rate × tokens/task** table. This is the honest re-run of
   the 0/15 and the make-or-break test of the whole thesis. Cheap on 8B.
2. **Phase 2 — visualization:** surface the controller in the web UI (thinking-budget
   meter, stability curve, stop reason). Portfolio hook + paper interpretability.
3. **README + assets:** embed the Phase 1.5 chart + honest numbers; record a ~15s GIF
   of the adaptive UI; re-pitch around adaptive compute.
4. **Deploy** on a free tier (Render / Fly / HF Spaces).
5. **Housekeeping:** (this branch is merged to `main` as of handoff).

**Boundary → "later scope" (needs investment $):** Phase 3 paper rig — real
benchmarks (GSM8K/MATH/graded QA), strong baselines (Self-Refine, Reflexion,
best-of-N) at matched compute, stronger generator, multi-seed error bars,
ablations, the accuracy-per-token Pareto figure. Optional Phase 4: cross-episode
experiential-memory transfer study. Full detail in `RESEARCH.md` §4.
