"""Evaluation harness: does recursive introspection beat single-shot?

Runs Ouroboros against a single-shot baseline on a fixed task set, scored by a
blind, position-swapped pairwise judge (a *different* free model than the
generator, to limit self-preference bias). See evals/README.md.
"""
