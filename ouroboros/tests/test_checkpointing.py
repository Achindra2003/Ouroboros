from __future__ import annotations

from unittest.mock import patch

import pytest

from langgraph.checkpoint.memory import MemorySaver

from ouroboros import checkpointing
from ouroboros.config import Settings


@pytest.mark.asyncio
async def test_memory_checkpointer_default():
    with patch.object(checkpointing, "get_settings", return_value=Settings(checkpointer="memory")):
        async with checkpointing.checkpointer_context() as cp:
            assert isinstance(cp, MemorySaver)


@pytest.mark.asyncio
async def test_explicit_kind_overrides_settings():
    with patch.object(checkpointing, "get_settings", return_value=Settings(checkpointer="sqlite")):
        async with checkpointing.checkpointer_context(kind="memory") as cp:
            assert isinstance(cp, MemorySaver)


@pytest.mark.asyncio
async def test_sqlite_checkpointer_persists_across_reopen(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    db = str(tmp_path / "ck.sqlite")
    settings = Settings(checkpointer="sqlite", sqlite_path=db)
    cfg = {"configurable": {"thread_id": "t1"}}

    class S(TypedDict):
        n: int

    async def inc(state):
        return {"n": state["n"] + 1}

    def build():
        b = StateGraph(S)
        b.add_node("inc", inc)
        b.add_edge(START, "inc")
        b.add_edge("inc", END)
        return b

    with patch.object(checkpointing, "get_settings", return_value=settings):
        async with checkpointing.checkpointer_context() as cp:
            g = build().compile(checkpointer=cp)
            await g.ainvoke({"n": 0}, config=cfg)
        # Reopen a fresh saver on the same path — state must persist.
        async with checkpointing.checkpointer_context() as cp2:
            g2 = build().compile(checkpointer=cp2)
            snap = await g2.aget_state(cfg)
    assert snap.values["n"] == 1
