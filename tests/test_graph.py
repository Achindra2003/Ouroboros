from __future__ import annotations

import pytest

from ouroboros.graph.builder import create_ouroboros_graph
from ouroboros.graph.nodes import (
    ingest,
    emotional_analysis,
    memory_search,
    synthesize,
    steer,
    should_use_tool,
    make_route_after_synthesis,
    make_route_after_breathe,
)
from ouroboros.graph.state import OuroborosState, extend_list
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
            "memory", "synthesize", "research_agent", "execute_tool",
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

    def test_emotional_analysis_with_keyword(self, base_state, default_config):
        base_state["thought"] = "I feel fear and uncertainty about this"
        result = emotional_analysis(base_state, default_config)
        assert result["mood"] == "anxious"
        assert "anxious" in result["emotional_reading"]

    def test_emotional_analysis_no_keyword(self, base_state, default_config):
        config = OuroborosConfig(mood_shift_chance=0)
        base_state["thought"] = "The weather is nice today"
        result = emotional_analysis(base_state, config)
        assert result["mood"] == "curious"

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

    def test_synthesize(self, base_state):
        base_state["emotional_reading"] = "Mood: curious"
        base_state["logical_reading"] = "Logic: sound"
        base_state["memory_reading"] = "Memory: connected"
        result = synthesize(base_state)
        assert "curious" in result["synthesis"]
        assert result["depth"] == 1

    @pytest.mark.asyncio
    async def test_surface(self, mock_llm, default_config, base_state):
        from ouroboros.graph.nodes import make_surface
        surface = make_surface(mock_llm, default_config)
        result = await surface(base_state)
        assert "surfaced_insight" in result
        assert "insights" in result

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
        router = make_route_after_synthesis(default_config)
        base_state["energy"] = 80
        base_state["depth"] = 1
        base_state["loop_guard"] = 0
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

    def test_should_use_tool_with_tool_calls(self, base_state):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "1"}])
        base_state["messages"] = [msg]
        assert should_use_tool(base_state) == "tools"

    def test_should_use_tool_without_tool_calls(self, base_state):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="No tools needed")
        base_state["messages"] = [msg]
        assert should_use_tool(base_state) == "done"

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
