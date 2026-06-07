from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ouroboros.graph.controller import Decision, answer_stability, decide
from ouroboros.graph.nodes import (
    _parse_confidence,
    make_route_after_breathe,
    make_route_after_synthesis,
    make_surface,
    make_synthesize,
)
from ouroboros.models import Mode, OuroborosConfig


def _adaptive_config(**kw) -> OuroborosConfig:
    base = dict(
        mode=Mode.EXPLORE,
        adaptive=True,
        min_cycles=1,
        compute_budget=3,
        stability_threshold=0.9,
        confidence_threshold=0.75,
    )
    base.update(kw)
    return OuroborosConfig(**base)


def _llm_returning(content: str):
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    llm.ainvoke = AsyncMock(return_value=resp)
    return llm


class TestParseConfidence:
    def test_parses_marker(self):
        ans, conf = _parse_confidence("A complete answer.\nCONFIDENCE: 0.9")
        assert ans == "A complete answer."
        assert conf == 0.9

    def test_defaults_when_missing(self):
        ans, conf = _parse_confidence("No marker here.")
        assert ans == "No marker here."
        assert conf == 0.5

    def test_clamps_out_of_range(self):
        # The regex only admits 0-1, so a stray "2" is ignored and we default.
        _, conf = _parse_confidence("x\nCONFIDENCE: 2.0")
        assert conf == 0.5

    def test_case_insensitive_and_stripped(self):
        ans, conf = _parse_confidence("Answer body.\nconfidence = 0.4")
        assert ans == "Answer body."
        assert conf == 0.4


class TestAnswerStability:
    def test_identical_is_high(self):
        assert answer_stability("the same answer", "the same answer") > 0.99

    def test_empty_prev_is_zero(self):
        assert answer_stability("", "anything") == 0.0

    def test_disjoint_is_low(self):
        # No shared tokens under the lexical fallback → near zero.
        assert answer_stability("alpha beta gamma", "delta epsilon zeta") < 0.5


class TestDecide:
    def _cfg(self):
        return _adaptive_config()

    def test_min_cycles_blocks_halt(self):
        d = decide(depth=0, stability=1.0, confidence=1.0, config=self._cfg())
        assert d == Decision(False, "min_cycles")

    def test_budget_forces_halt(self):
        d = decide(depth=3, stability=0.0, confidence=0.0, config=self._cfg())
        assert d.halt and d.reason == "budget"

    def test_converged(self):
        d = decide(depth=1, stability=0.95, confidence=0.8, config=self._cfg())
        assert d.halt and d.reason == "converged"

    def test_stable_but_unconfident_is_diminishing_returns(self):
        d = decide(depth=1, stability=0.95, confidence=0.1, config=self._cfg())
        assert d.halt and d.reason == "no_marginal_gain"

    def test_keep_going_when_unstable(self):
        d = decide(depth=1, stability=0.2, confidence=0.9, config=self._cfg())
        assert not d.halt and d.reason == "continue"


class TestAdaptiveSynthesize:
    @pytest.mark.asyncio
    async def test_emits_signals_and_strips_marker(self):
        cfg = _adaptive_config()
        llm = _llm_returning("A refined, complete answer.\nCONFIDENCE: 0.95")
        node = make_synthesize(llm, cfg)
        state = {"seed": "Q?", "synthesis": "old answer", "depth": 0}
        out = await node(state)
        assert out["synthesis"] == "A refined, complete answer."
        assert out["confidence"] == 0.95
        assert out["depth"] == 1
        assert "should_halt" in out and "stop_reason" in out
        assert out["prev_synthesis"] == "old answer"

    @pytest.mark.asyncio
    async def test_legacy_path_untouched_without_config(self):
        llm = _llm_returning("plain synthesis")
        node = make_synthesize(llm)  # no config → legacy behavior
        out = await node({"thought": "t", "depth": 0})
        assert out["synthesis"] == "plain synthesis"
        assert "confidence" not in out


class TestAdaptiveRouting:
    def test_halt_routes_to_surface(self):
        router = make_route_after_synthesis(_adaptive_config())
        assert router({"should_halt": True, "depth": 2}) == "surface"

    def test_continue_routes_to_think(self):
        router = make_route_after_synthesis(_adaptive_config(mode=Mode.EXPLORE))
        assert router({"should_halt": False, "depth": 2}) == "think"

    def test_breathe_ends_in_adaptive(self):
        router = make_route_after_breathe(_adaptive_config())
        assert router({}) == "__end__"


class TestAdaptiveSurface:
    @pytest.mark.asyncio
    async def test_passes_through_synthesis(self):
        cfg = _adaptive_config()
        # llm should NOT be called in adaptive surface; pass a guard that fails if it is.
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=AssertionError("surface must not call the LLM in adaptive mode"))
        node = make_surface(llm, cfg)
        out = await node({"synthesis": "the converged answer"})
        assert out["surfaced_insight"] == "the converged answer"
        assert out["insights"] == ["the converged answer"]
