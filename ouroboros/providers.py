"""LLM provider factory.

Single source of truth for constructing the chat model from settings, shared by
the CLI and the server. Supports three free-tier-friendly backends:

- ``groq``   — Groq's free API tier serving Llama models (default).
- ``ollama`` — fully local models via Ollama; zero API keys, runs offline.
- ``openai`` — OpenAI-compatible endpoints (also covers local OpenAI-style servers).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ouroboros.config import get_settings


def get_llm(temperature: float | None = None, model: str | None = None) -> BaseChatModel:
    """Build the configured chat model.

    Args:
        temperature: Override the configured temperature. When ``None`` the value
            from settings (``LLM_TEMPERATURE``) is used.
        model: Override the provider's configured model name (e.g. to use a
            different judge model in evaluations). When ``None`` the provider's
            default from settings is used.
    """
    settings = get_settings()
    temp = settings.llm_temperature if temperature is None else temperature
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.openai_model,
            temperature=temp,
            api_key=settings.openai_api_key or None,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'ollama' provider requires langchain-ollama. "
                "Install it with: pip install -e '.[local]'  (and run a local Ollama server)."
            ) from exc

        return ChatOllama(
            model=model or settings.ollama_model,
            temperature=temp,
            base_url=settings.ollama_base_url,
        )

    # Default: Groq (free tier, Llama models).
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model or settings.groq_model,
        temperature=temp,
        api_key=settings.groq_api_key or None,
    )
