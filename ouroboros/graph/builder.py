from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ouroboros.graph.state import OuroborosState
from ouroboros.graph.nodes import (
    ingest,
    make_think,
    make_reflect,
    make_emotional_analysis,
    make_logical_analysis,
    memory_search,
    make_synthesize,
    make_surface,
    remember,
    make_breathe,
    make_plan_research,
    make_research_worker,
    fan_out_research,
    steer,
    make_route_after_synthesis,
    make_route_after_breathe,
)
from ouroboros.models import OuroborosConfig


def create_ouroboros_graph(
    llm: BaseChatModel,
    config: OuroborosConfig | None = None,
    checkpointer=None,
):
    """Build the Ouroboros recursive introspection graph.

    Demonstrates these LangGraph patterns:
    - Fan-out / fan-in (parallel analysis: emotional, logical, memory → synthesize)
    - Send API map-reduce (plan_research → N parallel research_workers → reduce)
    - Conditional routing (route_after_synthesis, route_after_breathe, fan_out_research)
    - Human-in-the-loop (interrupt_before=["steer"])
    - Custom state reducers (extend_list for memories, insights, research findings)
    - Checkpointing (MemorySaver default, or durable AsyncSqliteSaver — pluggable)
    """
    if config is None:
        config = OuroborosConfig()

    builder = StateGraph(OuroborosState)

    builder.add_node("ingest", ingest)
    builder.add_node("think", make_think(llm, config))
    builder.add_node("reflect", make_reflect(llm, config))
    builder.add_node("emotional", make_emotional_analysis(llm, config))
    builder.add_node("logical", make_logical_analysis(llm))
    builder.add_node("memory", memory_search)
    builder.add_node("synthesize", make_synthesize(llm, config))
    builder.add_node("plan_research", make_plan_research(llm, config))
    builder.add_node("research_worker", make_research_worker())
    builder.add_node("surface", make_surface(llm, config))
    builder.add_node("remember", lambda s: remember(s, config))
    builder.add_node("breathe", make_breathe(config))
    builder.add_node("steer", steer)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "think")

    builder.add_edge("think", "reflect")

    builder.add_edge("reflect", "emotional")
    builder.add_edge("reflect", "logical")
    builder.add_edge("reflect", "memory")

    builder.add_edge("emotional", "synthesize")
    builder.add_edge("logical", "synthesize")
    builder.add_edge("memory", "synthesize")

    builder.add_conditional_edges(
        "synthesize",
        make_route_after_synthesis(config),
        {"think": "think", "research": "plan_research", "surface": "surface"},
    )

    # Send-based map-reduce: plan_research fans out to one worker per sub-query,
    # workers reduce their findings into state, then control returns to think.
    builder.add_conditional_edges("plan_research", fan_out_research, ["research_worker", "think"])
    builder.add_edge("research_worker", "think")

    builder.add_edge("surface", "remember")
    builder.add_edge("remember", "breathe")

    builder.add_conditional_edges(
        "breathe",
        make_route_after_breathe(config),
        {"steer": "steer", "think": "think", "__end__": END},
    )
    builder.add_edge("steer", "think")

    cp = checkpointer or MemorySaver()
    return builder.compile(checkpointer=cp, interrupt_before=["steer"])
