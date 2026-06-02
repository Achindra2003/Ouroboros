"""Token usage and cost accounting + optional LangSmith tracing.

Free to run: the default Groq tier and local Ollama cost nothing, so the
estimated cost is ``$0.00`` for them — but the same accounting surfaces real
numbers the moment someone points the project at a paid model. Tracing is
fully opt-in and off by default.
"""

from __future__ import annotations

import os

from langchain_core.callbacks import UsageMetadataCallbackHandler

from ouroboros.config import get_settings

# Approximate list prices in USD per 1M tokens (input, output).
# Free tiers (Groq, local Ollama) are modelled as $0; paid models use public rates.
# Matched by substring against the model name, longest match first.
_PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "llama-3.3-70b": (0.0, 0.0),   # Groq free tier
    "llama-3.1-8b": (0.0, 0.0),    # Groq free tier
    "llama-3.1-70b": (0.0, 0.0),   # Groq free tier
    "llama3": (0.0, 0.0),          # local Ollama
    "llama": (0.0, 0.0),           # local / free Llama fallback
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def new_usage_handler() -> UsageMetadataCallbackHandler:
    """Create a fresh callback handler that aggregates token usage for one run."""
    return UsageMetadataCallbackHandler()


def _price_for(model: str) -> tuple[float, float]:
    name = (model or "").lower()
    for key in sorted(_PRICES_PER_1M, key=len, reverse=True):
        if key in name:
            return _PRICES_PER_1M[key]
    return (0.0, 0.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a model given input/output token counts."""
    in_rate, out_rate = _price_for(model)
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def summarize_usage(handler: UsageMetadataCallbackHandler) -> dict:
    """Flatten a handler's per-model usage into totals + an estimated cost.

    Returns a JSON-serializable dict::

        {
          "input_tokens": int,
          "output_tokens": int,
          "total_tokens": int,
          "estimated_cost_usd": float,
          "by_model": {model: {input_tokens, output_tokens, total_tokens, cost_usd}},
        }
    """
    by_model: dict[str, dict] = {}
    total_in = total_out = total_all = 0
    total_cost = 0.0

    for model, usage in (handler.usage_metadata or {}).items():
        in_tok = int(usage.get("input_tokens", 0))
        out_tok = int(usage.get("output_tokens", 0))
        tot = int(usage.get("total_tokens", in_tok + out_tok))
        cost = estimate_cost(model, in_tok, out_tok)
        by_model[model] = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot,
            "cost_usd": round(cost, 6),
        }
        total_in += in_tok
        total_out += out_tok
        total_all += tot
        total_cost += cost

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_all,
        "estimated_cost_usd": round(total_cost, 6),
        "by_model": by_model,
    }


def configure_tracing() -> bool:
    """Enable LangSmith tracing if configured. Returns True if enabled.

    Opt-in via settings (``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY``).
    Exports the env vars LangChain reads natively, so the rest is automatic.
    """
    settings = get_settings()
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True
