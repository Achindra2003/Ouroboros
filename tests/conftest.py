from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ouroboros.models import Mode, OuroborosConfig


@pytest.fixture
def default_config():
    return OuroborosConfig()


@pytest.fixture
def analyze_config():
    return OuroborosConfig(mode=Mode.ANALYZE, max_depth=6, starting_energy=100)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "A test thought about the nature of recursion."
    llm.ainvoke = AsyncMock(return_value=response)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def base_state():
    return {
        "messages": [],
        "seed": "What is consciousness?",
        "thought": "What is consciousness?",
        "mood": "curious",
        "energy": 80.0,
        "depth": 0,
        "memories": [],
        "insights": [],
        "emotional_reading": "",
        "logical_reading": "",
        "memory_reading": "",
        "synthesis": "",
        "loop_guard": 0,
        "tick": 0,
        "surfaced_insight": "",
        "mode": "explore",
        "research_queries": [],
        "research_findings": [],
        "human_input": "",
        "steer_count": 0,
    }
