from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ouroboros.config import Settings
from ouroboros import providers


def _settings(**overrides) -> Settings:
    base = dict(llm_provider="groq", llm_temperature=0.7)
    base.update(overrides)
    return Settings(**base)


def test_get_llm_defaults_to_groq():
    with patch.object(providers, "get_settings", return_value=_settings()):
        with patch("langchain_groq.ChatGroq") as chat_groq:
            providers.get_llm()
    chat_groq.assert_called_once()
    assert chat_groq.call_args.kwargs["model"] == "llama-3.3-70b-versatile"


def test_get_llm_temperature_override_wins():
    with patch.object(providers, "get_settings", return_value=_settings(llm_temperature=0.1)):
        with patch("langchain_groq.ChatGroq") as chat_groq:
            providers.get_llm(temperature=0.9)
    assert chat_groq.call_args.kwargs["temperature"] == 0.9


def test_get_llm_uses_settings_temperature_when_none():
    with patch.object(providers, "get_settings", return_value=_settings(llm_temperature=0.33)):
        with patch("langchain_groq.ChatGroq") as chat_groq:
            providers.get_llm()
    assert chat_groq.call_args.kwargs["temperature"] == 0.33


def test_get_llm_openai_provider():
    with patch.object(providers, "get_settings", return_value=_settings(llm_provider="openai")):
        with patch("langchain_openai.ChatOpenAI") as chat_openai:
            providers.get_llm()
    chat_openai.assert_called_once()
    assert chat_openai.call_args.kwargs["model"] == "gpt-4o-mini"


def test_get_llm_ollama_provider():
    fake_module = MagicMock()
    with patch.object(providers, "get_settings", return_value=_settings(llm_provider="ollama")):
        with patch.dict("sys.modules", {"langchain_ollama": fake_module}):
            providers.get_llm()
    fake_module.ChatOllama.assert_called_once()
    assert fake_module.ChatOllama.call_args.kwargs["model"] == "llama3.2"


def test_get_llm_ollama_missing_dependency_raises_helpful_error():
    with patch.object(providers, "get_settings", return_value=_settings(llm_provider="ollama")):
        with patch.dict("sys.modules", {"langchain_ollama": None}):
            with pytest.raises(ImportError, match="langchain-ollama"):
                providers.get_llm()
