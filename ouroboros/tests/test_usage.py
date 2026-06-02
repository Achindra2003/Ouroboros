from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ouroboros import usage
from ouroboros.config import Settings


class _FakeHandler:
    def __init__(self, usage_metadata):
        self.usage_metadata = usage_metadata


def test_estimate_cost_free_model_is_zero():
    assert usage.estimate_cost("llama-3.3-70b-versatile", 1_000_000, 1_000_000) == 0.0


def test_estimate_cost_paid_model():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out
    cost = usage.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert round(cost, 4) == round(0.15 + 0.60, 4)


def test_estimate_cost_unknown_model_defaults_free():
    assert usage.estimate_cost("some-mystery-model", 5000, 5000) == 0.0


def test_summarize_usage_aggregates_models():
    handler = _FakeHandler({
        "llama-3.3-70b-versatile": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        "gpt-4o-mini": {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000},
    })
    summary = usage.summarize_usage(handler)
    assert summary["input_tokens"] == 1_000_100
    assert summary["output_tokens"] == 50
    assert summary["total_tokens"] == 1_000_150
    # Only the paid model contributes cost: 1M input * $0.15/1M = $0.15
    assert round(summary["estimated_cost_usd"], 4) == 0.15
    assert set(summary["by_model"]) == {"llama-3.3-70b-versatile", "gpt-4o-mini"}


def test_summarize_usage_empty_handler():
    summary = usage.summarize_usage(_FakeHandler({}))
    assert summary["total_tokens"] == 0
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["by_model"] == {}


def test_configure_tracing_disabled_by_default():
    with patch.object(usage, "get_settings", return_value=Settings(langsmith_tracing=False)):
        assert usage.configure_tracing() is False


def test_configure_tracing_enabled_sets_env(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test-key",
        langsmith_project="ouroboros-test",
    )
    with patch.object(usage, "get_settings", return_value=settings):
        assert usage.configure_tracing() is True
    import os
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "ouroboros-test"


def test_new_usage_handler_starts_empty():
    handler = usage.new_usage_handler()
    assert usage.summarize_usage(handler)["total_tokens"] == 0
    # sanity: object exposes the attribute we rely on
    assert hasattr(handler, "usage_metadata")
    _ = SimpleNamespace(ok=True)
