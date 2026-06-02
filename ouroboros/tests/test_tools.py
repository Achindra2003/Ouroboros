from __future__ import annotations

import os

import pytest

from ouroboros.graph.tools import web_search, retrieve_memories, ALL_TOOLS


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_no_api_key(self):
        old = os.environ.pop("TAVILY_API_KEY", None)
        try:
            result = await web_search.ainvoke({"query": "test query"})
            assert "unavailable" in result.lower()
        finally:
            if old:
                os.environ["TAVILY_API_KEY"] = old

    @pytest.mark.asyncio
    async def test_web_search_returns_string(self):
        old = os.environ.pop("TAVILY_API_KEY", None)
        try:
            result = await web_search.ainvoke({"query": "consciousness philosophy"})
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            if old:
                os.environ["TAVILY_API_KEY"] = old


class TestRetrieveMemories:
    @pytest.mark.asyncio
    async def test_retrieve_no_memories(self):
        result = await retrieve_memories.ainvoke({"query": "test", "memories": []})
        assert "no memories" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_with_related(self):
        mems = ["Recursive thinking enables self-awareness", "The weather is cloudy"]
        result = await retrieve_memories.ainvoke({"query": "recursive thinking", "memories": mems})
        assert "recursive" in result.lower() or "self-awareness" in result.lower()

    @pytest.mark.asyncio
    async def test_retrieve_unrelated_returns_recent(self):
        mems = ["First memory here", "Second memory there"]
        result = await retrieve_memories.ainvoke({"query": "xyzabc123 unrelated", "memories": mems})
        assert "recent" in result.lower()


class TestToolRegistry:
    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) >= 2
        names = [t.name for t in ALL_TOOLS]
        assert "web_search" in names
        assert "retrieve_memories" in names

    def test_tools_have_descriptions(self):
        for tool in ALL_TOOLS:
            assert tool.description
            assert len(tool.description) > 0
