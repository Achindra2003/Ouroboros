from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from ouroboros.graph.state import OuroborosState
from ouroboros.graph.nodes import (
    ingest,
    make_think,
    make_reflect,
    emotional_analysis,
    make_logical_analysis,
    memory_search,
    synthesize,
    make_surface,
    remember,
    make_breathe,
    make_research_agent,
    steer,
    should_use_tool,
    make_route_after_synthesis,
    make_route_after_breathe,
)
from ouroboros.graph.tools import ALL_TOOLS
from ouroboros.models import OuroborosConfig


def create_ouroboros_graph(
    llm: BaseChatModel,
    config: OuroborosConfig | None = None,
    checkpointer=None,
):
    """Build the Ouroboros recursive introspection graph.

    Demonstrates these LangGraph patterns:
    - Fan-out / fan-in (parallel analysis: emotional, logical, memory → synthesize)
    - ToolNode integration (research agent with web_search + retrieve_memories)
    - Conditional routing (route_after_synthesis, route_after_breathe, should_use_tool)
    - Human-in-the-loop (interrupt_before=["steer"])
    - Custom state reducers (extend_list for memories, insights, research queries)
    - Checkpointing (MemorySaver or pluggable checkpointer)
    """
    if config is None:
        config = OuroborosConfig()

    builder = StateGraph(OuroborosState)

    builder.add_node("ingest", ingest)
    builder.add_node("think", make_think(llm, config))
    builder.add_node("reflect", make_reflect(llm, config))
    builder.add_node("emotional", lambda s: emotional_analysis(s, config))
    builder.add_node("logical", make_logical_analysis(llm))
    builder.add_node("memory", memory_search)
    builder.add_node("synthesize", synthesize)
    builder.add_node("research_agent", make_research_agent(llm, config))
    builder.add_node("execute_tool", ToolNode(ALL_TOOLS))
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
        {"think": "think", "research": "research_agent", "surface": "surface"},
    )

    builder.add_conditional_edges(
        "research_agent",
        should_use_tool,
        {"tools": "execute_tool", "done": "think"},
    )
    builder.add_edge("execute_tool", "research_agent")

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
