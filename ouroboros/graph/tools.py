from __future__ import annotations

import os

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information about a topic. Returns relevant snippets."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return (
            f"[Web search unavailable for: '{query}'. "
            "Set TAVILY_API_KEY to enable live web search.]"
        )
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        results = client.search(query, max_results=3)
        snippets = []
        for r in results.get("results", [])[:3]:
            snippets.append(f"- {r.get('title', '')}: {r.get('content', '')}")
        return "\n".join(snippets) if snippets else f"No results found for: {query}"
    except ImportError:
        return (
            f"[Web search unavailable for: '{query}'. "
            "Install tavily-python to enable live search.]"
        )
    except Exception as e:
        return f"[Search error for '{query}': {e}]"


@tool
def retrieve_memories(query: str, memories: list[str] | None = None) -> str:
    """Search through stored memories for connections to a query."""
    from ouroboros.memory import semantic_search

    mems = memories or []
    if not mems:
        return "No memories stored yet."
    results = semantic_search(query, mems, k=3)
    related = [m for m, score in results if score > 0]
    if related:
        return "Related memories: " + "; ".join(related)
    return "No strongly connected memories. Recent: " + "; ".join(mems[-2:])


ALL_TOOLS = [web_search, retrieve_memories]
