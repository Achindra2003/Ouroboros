from __future__ import annotations

import pytest

from ouroboros.graph.builder import create_ouroboros_graph
from ouroboros.graph.nodes import (
    ingest,
    derive_mood,
    make_emotional_analysis,
    memory_search,
    make_synthesize,
    make_plan_research,
    make_research_worker,
    fan_out_research,
    steer,
    make_route_after_synthesis,
    make_route_after_breathe,
)
from ouroboros.graph.state import extend_list
from ouroboros.models import Mode, OuroborosConfig


class TestCustomReducers:
    def test_extend_list_appends(self):
        result = extend_list(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_extend_list_deduplicates(self):
        result = extend_list(["a", "b"], ["b", "c"])
        assert result == ["a", "b", "c"]

    def test_extend_list_none_update(self):
        result = extend_list(["a"], None)
        assert result == ["a"]

    def test_extend_list_empty_current(self):
        result = extend_list([], ["a", "b"])
        assert result == ["a", "b"]


class TestGraphCreation:
    def test_creates_graph_with_default_config(self, mock_llm):
        graph = create_ouroboros_graph(mock_llm)
        assert graph is not None

    def test_creates_graph_for_each_mode(self, mock_llm):
        for mode in Mode:
            config = OuroborosConfig(mode=mode)
            graph = create_ouroboros_graph(mock_llm, config)
            assert graph is not None

    def test_graph_has_expected_nodes(self, mock_llm, default_config):
        graph = create_ouroboros_graph(mock_llm, default_config)
        node_names = set(graph.nodes.keys())
        expected = {
            "ingest", "think", "reflect", "emotional", "logical",
            "memory", "synthesize", "plan_research", "research_worker",
            "surface", "remember", "breathe", "steer",
        }
        assert expected.issubset(node_names)


class TestNodeFunctions:
    @pytest.mark.asyncio
    async def test_ingest(self, base_state):
        result = ingest(base_state)
        assert result["thought"] == base_state["seed"]
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_think(self, mock_llm, default_config, base_state):
        from ouroboros.graph.nodes import make_think
        think = make_think(mock_llm, default_config)
        result = await think(base_state)
        assert "thought" in result
        assert "messages" in result
        assert result["energy"] < base_state["energy"]

    @pytest.mark.asyncio
    async def test_reflect(self, mock_llm, default_config, base_state):
        from ouroboros.graph.nodes import make_reflect
        reflect = make_reflect(mock_llm, default_config)
        result = await reflect(base_state)
        assert "messages" in result

    def test_derive_mood_with_keyword(self, default_config):
        assert derive_mood("I feel fear and uncertainty about this", "curious", default_config) == "anxious"

    def test_derive_mood_no_keyword_no_shift(self):
        config = OuroborosConfig(mood_shift_chance=0)
        assert derive_mood("The weather is nice today", "curious", config) == "curious"

    @pytest.mark.asyncio
    async def test_emotional_analysis_node_uses_llm_and_derives_mood(
        self, mock_llm, default_config, base_state
    ):
        node = make_emotional_analysis(mock_llm, default_config)
        base_state["thought"] = "I feel fear and uncertainty"
        result = await node(base_state)
        assert result["mood"] == "anxious"  # keyword prior still drives routing
        assert result["emotional_reading"]  # richer reading comes from the LLM

    @pytest.mark.asyncio
    async def test_emotional_analysis_node_falls_back_on_error(
        self, mock_llm, default_config, base_state
    ):
        from unittest.mock import AsyncMock

        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("rate limited"))
        node = make_emotional_analysis(mock_llm, default_config)
        base_state["thought"] = "peace and acceptance"
        result = await node(base_state)
        assert result["mood"] == "serene"
        assert "serene" in result["emotional_reading"]

    def test_memory_search_with_related(self, base_state):
        base_state["memories"] = [
            "Consciousness arises from recursive self-modeling",
            "The weather affects mood significantly",
        ]
        base_state["thought"] = "What is recursive self-modeling?"
        result = memory_search(base_state)
        assert "recursive" in result["memory_reading"].lower() or "self-modeling" in result["memory_reading"].lower()

    def test_memory_search_no_memories(self, base_state):
        base_state["memories"] = []
        result = memory_search(base_state)
        assert "unanchored" in result["memory_reading"]

    @pytest.mark.asyncio
    async def test_synthesize_integrates_via_llm(self, mock_llm, base_state):
        node = make_synthesize(mock_llm)
        base_state["emotional_reading"] = "Mood: curious"
        base_state["logical_reading"] = "Logic: sound"
        base_state["memory_reading"] = "Memory: connected"
        result = await node(base_state)
        assert result["synthesis"]  # integrated reading from the LLM
        assert result["depth"] == 1

    @pytest.mark.asyncio
    async def test_synthesize_fallback_concatenates_on_error(self, mock_llm, base_state):
        from unittest.mock import AsyncMock

        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        node = make_synthesize(mock_llm)
        base_state["emotional_reading"] = "EMO"
        base_state["logical_reading"] = "LOG"
        base_state["memory_reading"] = "MEM"
        result = await node(base_state)
        assert "EMO" in result["synthesis"]
        assert "LOG" in result["synthesis"]
        assert result["depth"] == 1

    @pytest.mark.asyncio
    async def test_surface(self, mock_llm, default_config, base_state):
        from ouroboros.graph.nodes import make_surface
        surface = make_surface(mock_llm, default_config)
        result = await surface(base_state)
        assert "surfaced_insight" in result
        assert "insights" in result

    @pytest.mark.asyncio
    async def test_plan_research_produces_queries(self, mock_llm, default_config, base_state):
        node = make_plan_research(mock_llm, default_config)
        result = await node(base_state)
        assert result["pending_queries"], "should plan at least one query"
        assert result["research_queries"] == result["pending_queries"]

    @pytest.mark.asyncio
    async def test_plan_research_falls_back_on_llm_error(self, mock_llm, default_config, base_state):
        # Free-tier rate limits / network errors must not crash the graph.
        from unittest.mock import AsyncMock

        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("429 rate limited"))
        node = make_plan_research(mock_llm, default_config)
        result = await node(base_state)
        assert result["pending_queries"], "fallback must still yield a query"

    @pytest.mark.asyncio
    async def test_research_worker_returns_finding(self):
        node = make_research_worker()
        result = await node({"query": "what is recursion"})
        assert result["research_findings"]
        assert "what is recursion" in result["research_findings"][0]

    def test_fan_out_research_emits_sends(self, base_state):
        from langgraph.types import Send

        base_state["pending_queries"] = ["q1", "q2", "q3"]
        sends = fan_out_research(base_state)
        assert isinstance(sends, list) and len(sends) == 3
        assert all(isinstance(s, Send) and s.node == "research_worker" for s in sends)

    def test_fan_out_research_skips_to_think_when_empty(self, base_state):
        base_state["pending_queries"] = []
        assert fan_out_research(base_state) == "think"

    def test_steer_with_input(self, base_state):
        base_state["human_input"] = "Dig deeper into recursion"
        result = steer(base_state)
        assert result["thought"] == "Dig deeper into recursion"
        assert result["human_input"] == ""

    def test_steer_without_input(self, base_state):
        result = steer(base_state)
        assert result["steer_count"] == 1


class TestRouting:
    def test_route_after_synthesis_continue(self, base_state, default_config):
        # The router has a stochastic ~20% branch to "research"; pin the RNG above
        # that threshold so this test deterministically exercises the "think" path.
        from unittest.mock import patch

        router = make_route_after_synthesis(default_config)
        base_state["energy"] = 80
        base_state["depth"] = 1
        base_state["loop_guard"] = 0
        with patch("ouroboros.graph.nodes.random.random", return_value=0.99):
            assert router(base_state) == "think"

    def test_route_after_synthesis_surface_low_energy(self, base_state, default_config):
        router = make_route_after_synthesis(default_config)
        base_state["energy"] = 5
        assert router(base_state) == "surface"

    def test_route_after_synthesis_surface_max_depth(self, base_state, default_config):
        router = make_route_after_synthesis(default_config)
        base_state["energy"] = 80
        base_state["depth"] = default_config.max_depth
        assert router(base_state) == "surface"

    def test_route_after_synthesis_surface_serene(self, base_state, default_config):
        router = make_route_after_synthesis(default_config)
        base_state["energy"] = 80
        base_state["depth"] = 3
        base_state["mood"] = "serene"
        assert router(base_state) == "surface"

    def test_route_after_breathe_continue(self, base_state, default_config):
        router = make_route_after_breathe(default_config)
        base_state["energy"] = 80
        base_state["loop_guard"] = 1
        base_state["steer_count"] = 0
        assert router(base_state) == "think"

    def test_route_after_breathe_end(self, base_state, default_config):
        router = make_route_after_breathe(default_config)
        base_state["energy"] = 10
        base_state["loop_guard"] = 16
        assert router(base_state) == "__end__"

    def test_route_after_breathe_steer(self, base_state, default_config):
        router = make_route_after_breathe(default_config)
        base_state["energy"] = 80
        base_state["loop_guard"] = 3
        base_state["steer_count"] = 0
        assert router(base_state) == "steer"
