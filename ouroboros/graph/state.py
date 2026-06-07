from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def extend_list(current: list, update: list | None) -> list:
    if update is None:
        return current
    return current + [u for u in update if u not in current]


def replace_value(current, update):
    return update if update is not None else current


class OuroborosState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    seed: str
    thought: str
    mood: str
    energy: float
    depth: int
    memories: Annotated[list[str], extend_list]
    insights: Annotated[list[str], extend_list]
    emotional_reading: str
    logical_reading: str
    memory_reading: str
    synthesis: str
    loop_guard: int
    tick: int
    surfaced_insight: str
    mode: str
    research_queries: Annotated[list[str], extend_list]
    research_findings: Annotated[list[str], extend_list]
    pending_queries: list[str]  # current batch to fan out (replace semantics)
    human_input: str
    steer_count: int
    # --- metacognitive adaptive controller (Wedge A); populated only when
    # config.adaptive is on. All replace-semantics, written solely by synthesize.
    prev_synthesis: str  # previous cycle's answer, for stability measurement
    confidence: float  # synthesizer's self-reported settledness (0-1)
    stability: float  # cosine similarity between this and the previous answer
    should_halt: bool  # controller's halt decision (read by the router)
    stop_reason: str  # why the controller halted (converged | budget | ...)
