"""Run the two systems under comparison: Ouroboros vs. a single-shot baseline.

Both use the same generation model so the comparison isolates the *method*
(recursive multi-perspective introspection) rather than model strength.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from evals.llm_utils import ainvoke_text
from ouroboros.graph import create_ouroboros_graph
from ouroboros.models import Mode
from ouroboros.presets import MODE_PRESETS

# A deliberately strong single-shot prompt — the baseline should be fair, not a
# strawman. It is allowed to reason and is asked for a non-obvious, crystallized take.
_BASELINE_PROMPT = (
    "You are a thoughtful mind answering in {mode} mode.\n\n"
    'Question: "{seed}"\n\n'
    "Think carefully, then give your single most insightful, non-obvious response. "
    "Avoid platitudes. 2-4 sentences."
)


def _initial_state(seed: str, mode_value: str, starting_energy: float) -> dict:
    return {
        "messages": [],
        "thought": seed,
        "mood": "curious",
        "energy": starting_energy,
        "depth": 0,
        "memories": [],
        "insights": [],
        "emotional_reading": "",
        "logical_reading": "",
        "memory_reading": "",
        "synthesis": "",
        "loop_guard": 0,
        "tick": 0,
        "seed": seed,
        "surfaced_insight": "",
        "mode": mode_value,
        "research_queries": [],
        "research_findings": [],
        "human_input": "",
        "steer_count": 0,
    }


async def run_ouroboros(
    llm: BaseChatModel, seed: str, mode: str, max_cycles: int
) -> str:
    """Run a full introspection session headlessly; return the surfaced insight(s)."""
    mode_enum = Mode(mode)
    config = MODE_PRESETS[mode_enum]["config"].model_copy(
        update={"max_loop_guard": max_cycles, "steer_interval": 999}
    )
    graph = create_ouroboros_graph(llm, config)
    state = _initial_state(seed, config.mode.value, config.starting_energy)
    graph_config = {"configurable": {"thread_id": f"eval-{seed[:24]}"}}

    insights: list[str] = []
    async for event in graph.astream(state, config=graph_config, stream_mode="updates"):
        update = event[next(iter(event))]
        if isinstance(update, dict) and update.get("surfaced_insight"):
            insights.append(update["surfaced_insight"])

    if not insights:
        return "(no insight surfaced)"
    # Use the final, most-refined insight as the answer under test.
    return insights[-1]


async def run_baseline(llm: BaseChatModel, seed: str, mode: str) -> str:
    """Single-shot baseline: one LLM call answering the seed directly."""
    prompt = _BASELINE_PROMPT.format(mode=mode, seed=seed)
    return await ainvoke_text(llm, prompt)
