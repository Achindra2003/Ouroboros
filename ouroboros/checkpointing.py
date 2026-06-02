"""Checkpointer factory — honors the `checkpointer` / `sqlite_path` settings.

LangGraph checkpoints let a run be persisted and resumed (including across process
restarts). Two backends:

- ``memory`` — in-process MemorySaver (default; fine for a single live run).
- ``sqlite`` — durable AsyncSqliteSaver: a session keyed by ``thread_id`` survives
  restarts. Requires the ``persist`` extra (``pip install -e '.[persist]'``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver

from ouroboros.config import get_settings


@asynccontextmanager
async def checkpointer_context(kind: str | None = None):
    """Yield a checkpointer for the duration of a run.

    Used as ``async with checkpointer_context() as cp: ...`` so the durable
    sqlite connection is opened and closed around the graph execution.
    """
    settings = get_settings()
    kind = kind or settings.checkpointer

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "sqlite checkpointing requires the persist extra: "
                "pip install -e '.[persist]'"
            ) from exc
        async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as cp:
            yield cp
    else:
        yield MemorySaver()
