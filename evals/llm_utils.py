"""Resilient direct-call helper for evaluations.

Free tiers rate-limit aggressively, so direct LLM calls (baseline + judge) retry
with exponential backoff + jitter. The Ouroboros graph is intentionally driven by
the raw model instead (it needs ``bind_tools`` and already degrades gracefully
per-node), so this helper is only for the single-call paths.
"""

from __future__ import annotations

import asyncio
import random

from langchain_core.language_models import BaseChatModel


async def ainvoke_text(
    llm: BaseChatModel,
    prompt: str,
    attempts: int = 5,
    base_delay: float = 2.0,
) -> str:
    """Invoke ``llm`` with a system prompt, retrying transient failures."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            return resp.content.strip()
        except Exception as exc:  # rate limit, transient network, etc.
            last_exc = exc
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i) + random.random())
    raise last_exc  # type: ignore[misc]
