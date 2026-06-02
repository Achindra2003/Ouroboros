from __future__ import annotations

import pytest

from evals import judge as judge_mod
from evals.engine import run_baseline
from evals.judge import _parse_verdict, compare
from evals.run import _summarize
from evals.tasks import get_tasks


class _ScriptedLLM:
    """Async chat model stub that returns queued contents in order."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    async def ainvoke(self, messages):
        from types import SimpleNamespace

        content = self._replies[self.calls % len(self._replies)]
        self.calls += 1
        return SimpleNamespace(content=content)


class TestParseVerdict:
    def test_parse_one(self):
        assert _parse_verdict("1") == "1"

    def test_parse_two_with_noise(self):
        assert _parse_verdict("Answer 2 is better") == "2"

    def test_parse_tie(self):
        assert _parse_verdict("tie") == "tie"

    def test_parse_unparseable_defaults_tie(self):
        assert _parse_verdict("hmm, hard to say") == "tie"


class TestCompare:
    @pytest.mark.asyncio
    async def test_consistent_ouroboros_win(self):
        # Order A (ouro=1): says "1". Order B (ouro=2): says "2". Consistent -> ouroboros.
        llm = _ScriptedLLM(["1", "2"])
        assert await compare(llm, "seed", "ouro answer", "base answer") == "ouroboros"

    @pytest.mark.asyncio
    async def test_consistent_baseline_win(self):
        # Order A (ouro=1): says "2". Order B (ouro=2): says "1". Consistent -> baseline.
        llm = _ScriptedLLM(["2", "1"])
        assert await compare(llm, "seed", "ouro answer", "base answer") == "baseline"

    @pytest.mark.asyncio
    async def test_inconsistent_is_tie(self):
        # Judge picks position 1 both times -> position bias -> tie.
        llm = _ScriptedLLM(["1", "1"])
        assert await compare(llm, "seed", "ouro answer", "base answer") == "tie"


class TestSummarize:
    def test_summarize_counts_and_rates(self):
        results = [
            {"mode": "analyze", "winner": "ouroboros"},
            {"mode": "analyze", "winner": "baseline"},
            {"mode": "solve", "winner": "ouroboros"},
            {"mode": "solve", "winner": "tie"},
        ]
        s = _summarize(results)
        assert s["total"] == 4
        assert s["ouroboros_wins"] == 2
        assert s["baseline_wins"] == 1
        assert s["ties"] == 1
        assert s["ouroboros_win_rate"] == 0.5
        assert s["by_mode"]["analyze"] == {"ouroboros": 1, "baseline": 1}


class TestEngineBaseline:
    @pytest.mark.asyncio
    async def test_run_baseline_returns_text(self):
        llm = _ScriptedLLM(["  a crisp single-shot answer  "])
        out = await run_baseline(llm, "What is X?", "explore")
        assert out == "a crisp single-shot answer"


class TestTasks:
    def test_get_tasks_limit(self):
        assert len(get_tasks(3)) == 3

    def test_get_tasks_all_have_valid_modes(self):
        valid = {"explore", "analyze", "create", "solve", "philosophize"}
        assert all(t.mode in valid for t in get_tasks())


def test_judge_module_exposes_compare():
    # guards against accidental rename of the public entry point
    assert hasattr(judge_mod, "compare")
