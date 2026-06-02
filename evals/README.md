# Evaluation: does recursion actually beat single-shot?

This harness tests the project's core claim — that recursive multi-perspective
introspection produces better output than a single prompt — instead of just
asserting it.

## Method

For each task in a fixed [`tasks.py`](tasks.py) set (spanning all five modes):

1. **Ouroboros** runs a full introspection session and surfaces an insight.
2. A **single-shot baseline** answers the same seed with one LLM call. Both use
   the **same generation model**, so the comparison isolates the *method*, not
   model strength. The baseline prompt is deliberately strong (not a strawman).
3. A **judge** scores the two answers on insight depth, non-obviousness,
   groundedness, and illumination.

### Bias controls

- The **judge is a different model** than the generator (set `EVAL_JUDGE_MODEL`),
  so it doesn't just prefer its own style.
- Answers are shown **blind** ("Answer 1" / "Answer 2").
- Each pair is judged **twice with positions swapped**; a win counts only if the
  judge is **consistent** across both orderings — otherwise it's a tie. This
  cancels position bias and softens self-preference.

## Running it (free)

```bash
pip install -e ".[eval]"          # adds matplotlib
# set GROQ_API_KEY in .env (free tier). For an unbiased judge, use a different model:
export EVAL_JUDGE_MODEL=llama-3.1-8b-instant   # or any other free model

python -m evals run --limit 5     # smoke test on 5 tasks
python -m evals run               # full task set
python -m evals plot              # writes evals/results/chart.png
```

Results and an LLM response cache are written to `evals/results/` (cache makes
re-runs cheap and reproducible). Configure via env: `EVAL_GEN_MODEL`,
`EVAL_JUDGE_MODEL`, `EVAL_GEN_TEMPERATURE`, `EVAL_MAX_CYCLES`, `EVAL_OUT_DIR`.

## Honesty note

Report the numbers you get — including modes where recursion *doesn't* win. A
credible "it wins on analyze/solve, ties on explore" is worth more than an
unbelievable clean sweep.
