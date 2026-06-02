"""Blind, position-swapped pairwise judge.

Bias controls (all free):
- The judge is a *different* model than the generator (configurable).
- Answers are presented blind as "Answer 1" / "Answer 2".
- Each pair is judged twice with positions swapped; a win counts only if the
  judge is consistent across both orderings, otherwise it's a tie. This cancels
  position bias and softens self-preference.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel

from evals.llm_utils import ainvoke_text

_RUBRIC = (
    "insight depth, non-obviousness, groundedness (not vague platitudes), "
    "and how much it actually illuminates the question"
)

_JUDGE_PROMPT = (
    "You are evaluating two answers to the same question. Judge ONLY on: "
    f"{_RUBRIC}.\n\n"
    'Question: "{seed}"\n\n'
    "Answer 1:\n{answer_1}\n\n"
    "Answer 2:\n{answer_2}\n\n"
    "Which answer is better? Reply with EXACTLY one token: \"1\", \"2\", or \"tie\". "
    "Do not explain."
)


def _parse_verdict(text: str) -> str:
    t = text.strip().lower()
    if re.search(r"\btie\b", t):
        return "tie"
    # Prefer the first standalone digit.
    m = re.search(r"[12]", t)
    if m:
        return m.group(0)
    return "tie"


async def _judge_once(
    judge_llm: BaseChatModel, seed: str, answer_1: str, answer_2: str
) -> str:
    prompt = _JUDGE_PROMPT.format(seed=seed, answer_1=answer_1, answer_2=answer_2)
    text = await ainvoke_text(judge_llm, prompt)
    return _parse_verdict(text)


async def compare(
    judge_llm: BaseChatModel, seed: str, ouroboros: str, baseline: str
) -> str:
    """Return the consistent winner across both orderings.

    Result is one of: "ouroboros", "baseline", "tie".
    """
    # Order A: ouroboros = position 1, baseline = position 2.
    v1 = await _judge_once(judge_llm, seed, ouroboros, baseline)
    # Order B: swapped — baseline = position 1, ouroboros = position 2.
    v2 = await _judge_once(judge_llm, seed, baseline, ouroboros)

    ouro_wins = (v1 == "1") and (v2 == "2")
    base_wins = (v1 == "2") and (v2 == "1")
    if ouro_wins:
        return "ouroboros"
    if base_wins:
        return "baseline"
    return "tie"
