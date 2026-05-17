from __future__ import annotations

import random

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from ouroboros.graph.state import OuroborosState
from ouroboros.models import Mode, OuroborosConfig
from ouroboros.presets import MODE_PRESETS


FALLBACK_THOUGHTS = {
    "curious": "What lies beneath this thought? The surface is only the beginning.",
    "anxious": "The thought trembles. What am I afraid of seeing?",
    "obsessed": "I cannot stop returning to this. Why does it grip me so?",
    "melancholic": "The thought fades like a memory. What was it I lost?",
    "serene": "The thought settles. There is nothing more to chase.",
    "ecstatic": "The thought blazes! Everything connects!",
}

FALLBACK_REFLECTIONS = [
    "The thought loops back on itself. What am I not seeing?",
    "There is a pattern here — the mind returns to the same place.",
    "What if the question IS the answer?",
    "The harder I look, the more the thought dissolves.",
    "Something hides beneath the surface of this thought.",
]

FALLBACK_LOGICAL = [
    "The logic is circular — the thought feeds on itself.",
    "An assumption hides beneath: that thinking will lead somewhere.",
    "The thought assumes its own importance. What if it doesn't matter?",
    "There is a gap in the reasoning — what was left unsaid?",
]

FALLBACK_INSIGHTS = [
    "After all this, the thought returns to where it began — but changed.",
    "The rumination reveals: the question was never the point.",
    "What I found: the act of looking is itself the discovery.",
    "The loop closes. What was sought was the seeker all along.",
]

MOOD_KEYWORDS = {
    "anxious": ["fear", "worry", "uncertain", "lost", "danger", "trembling", "shake", "avoid"],
    "obsessed": ["must", "always", "never", "cannot stop", "return", "grip", "again", "repeat"],
    "melancholic": ["fade", "end", "memory", "lost", "was", "sorrow", "empty", "gone"],
    "serene": ["peace", "accept", "still", "enough", "quiet", "settle", "rest", "calm"],
    "ecstatic": ["yes", "found", "beauty", "wonder", "alive", "blaze", "light", "joy"],
}

MOOD_READINGS = {
    "anxious": "The thought trembles with uncertainty.",
    "obsessed": "The mind grips tightly, unable to release.",
    "melancholic": "A sadness permeates the thought.",
    "serene": "The thought floats in stillness.",
    "ecstatic": "The thought blazes with discovery.",
    "curious": "The thought reaches outward with wonder.",
}

MOOD_SHIFTS = {
    "curious": ["wonder", "obsessed", "anxious"],
    "anxious": ["obsessed", "melancholic", "curious"],
    "obsessed": ["melancholic", "serene", "anxious"],
    "melancholic": ["serene", "curious", "anxious"],
    "serene": ["curious", "ecstatic", "melancholic"],
    "ecstatic": ["obsessed", "serene", "anxious"],
}


def _get_prompt(mode: Mode, key: str) -> str:
    preset = MODE_PRESETS.get(mode, MODE_PRESETS[Mode.EXPLORE])
    return preset.get(key, "")


def ingest(state: OuroborosState) -> dict:
    seed = state.get("seed", "What am I?")
    return {
        "thought": seed,
        "messages": [AIMessage(content=f"[seed] {seed}")],
    }


def make_think(llm: BaseChatModel, config: OuroborosConfig):
    prompt_template = _get_prompt(config.mode, "think_prompt")

    async def think(state: OuroborosState) -> dict:
        recent = [m.content for m in state["messages"][-6:] if isinstance(m, AIMessage)]
        mems = state.get("memories", [])[-5:]
        prompt = prompt_template.format(
            mood=state.get("mood", "curious"),
            depth=state.get("depth", 0),
            recent="\n".join(recent[-3:]) or "(beginning)",
            memories="\n".join(mems) or "(no memories yet)",
            seed=state.get("seed", ""),
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            new_thought = resp.content.strip()
        except Exception:
            new_thought = FALLBACK_THOUGHTS.get(
                state.get("mood", "curious"), FALLBACK_THOUGHTS["curious"]
            )
        return {
            "thought": new_thought,
            "messages": [AIMessage(content=new_thought)],
            "energy": state.get("energy", config.starting_energy) - config.energy_drain_think,
            "tick": state.get("tick", 0) + 1,
        }

    return think


def make_reflect(llm: BaseChatModel, config: OuroborosConfig):
    prompt_template = _get_prompt(config.mode, "reflect_prompt")

    async def reflect(state: OuroborosState) -> dict:
        prompt = prompt_template.format(
            thought=state.get("thought", ""),
            mood=state.get("mood", "curious"),
            seed=state.get("seed", ""),
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            reflection = resp.content.strip()
        except Exception:
            reflection = random.choice(FALLBACK_REFLECTIONS)
        return {
            "messages": [AIMessage(content=reflection)],
            "energy": state.get("energy", config.starting_energy)
            - config.energy_drain_reflect,
        }

    return reflect


def emotional_analysis(state: OuroborosState, config: OuroborosConfig = None) -> dict:
    if config is None:
        config = OuroborosConfig()
    thought = state.get("thought", "")
    mood = state.get("mood", "curious")
    thought_lower = thought.lower()
    new_mood = mood
    for candidate, words in MOOD_KEYWORDS.items():
        if any(w in thought_lower for w in words):
            new_mood = candidate
            break
    else:
        if random.random() < config.mood_shift_chance:
            new_mood = random.choice(MOOD_SHIFTS.get(mood, ["curious"]))
    reading = f"Emotional undertone: {new_mood}. {MOOD_READINGS.get(new_mood, '')}"
    return {"mood": new_mood, "emotional_reading": reading}


def make_logical_analysis(llm: BaseChatModel):
    async def logical_analysis(state: OuroborosState) -> dict:
        prompt = (
            f'Examine this thought for coherence: "{state.get("thought", "")}"\n\n'
            "What is unexamined? What assumption is unchallenged? "
            "What is the blind spot?\n\nOne sentence."
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            reading = resp.content.strip()
        except Exception:
            reading = random.choice(FALLBACK_LOGICAL)
        return {"logical_reading": reading}

    return logical_analysis


def memory_search(state: OuroborosState) -> dict:
    thought = state.get("thought", "")
    mems = state.get("memories", [])
    thought_words = set(thought.lower().split())
    related = [m for m in mems if len(thought_words & set(m.lower().split())) > 1]
    if not related and mems:
        related = mems[-2:]
    reading = (
        ("Connected memories: " + "; ".join(related[-3:]))
        if related
        else "No connected memories. The thought is unanchored."
    )
    return {"memory_reading": reading}


def synthesize(state: OuroborosState) -> dict:
    emo = state.get("emotional_reading", "")
    logic = state.get("logical_reading", "")
    mem = state.get("memory_reading", "")
    return {
        "synthesis": f"{emo} {logic} {mem}",
        "depth": state.get("depth", 0) + 1,
    }


def make_surface(llm: BaseChatModel, config: OuroborosConfig):
    prompt_template = _get_prompt(config.mode, "surface_prompt")

    async def surface(state: OuroborosState) -> dict:
        prompt = prompt_template.format(
            depth=state.get("depth", 1),
            seed=state.get("seed", ""),
            thought=state.get("thought", ""),
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            insight = resp.content.strip()
        except Exception:
            insight = random.choice(FALLBACK_INSIGHTS)
        return {
            "surfaced_insight": insight,
            "messages": [AIMessage(content=f"[insight] {insight}")],
            "insights": [insight],
        }

    return surface


def remember(state: OuroborosState, config: OuroborosConfig = None) -> dict:
    if config is None:
        config = OuroborosConfig()
    insight = state.get("surfaced_insight", state.get("thought", ""))
    mems = list(state.get("memories", []))
    if len(mems) >= config.max_memories:
        compressed = "Earlier: " + "; ".join(mems[:4])[:200]
        mems = [compressed] + mems[4:]
    mems.append(insight[:200])
    return {"memories": mems[-config.max_memories:]}


def make_breathe(config: OuroborosConfig):
    def breathe(state: OuroborosState) -> dict:
        energy = min(
            100, state.get("energy", 0) + config.energy_recovery_breathe
        )
        mood = state.get("mood", "curious")
        if mood in ("anxious", "obsessed"):
            mood = "curious" if random.random() < 0.6 else mood
        return {
            "energy": energy,
            "mood": mood,
            "depth": 0,
            "loop_guard": state.get("loop_guard", 0) + 1,
        }

    return breathe


def make_research_agent(llm: BaseChatModel, config: OuroborosConfig):
    from ouroboros.graph.tools import ALL_TOOLS

    prompt_template = _get_prompt(config.mode, "research_prompt")
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def research_agent(state: OuroborosState) -> dict:
        prompt = prompt_template.format(
            seed=state.get("seed", ""),
            thought=state.get("thought", ""),
        )
        messages = [SystemMessage(content=prompt)] + state["messages"][-6:]
        response = await llm_with_tools.ainvoke(messages)
        queries = list(state.get("research_queries", []))
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                queries.append(tc.get("args", {}).get("query", str(tc)))
        return {
            "messages": [response],
            "tick": state.get("tick", 0) + 1,
            "research_queries": queries,
        }

    return research_agent


def steer(state: OuroborosState) -> dict:
    human_input = state.get("human_input", "")
    if human_input:
        return {
            "messages": [AIMessage(content=f"[steer] {human_input}")],
            "thought": human_input,
            "human_input": "",
            "steer_count": state.get("steer_count", 0) + 1,
            "energy": min(100, state.get("energy", 0) + 15),
        }
    return {"steer_count": state.get("steer_count", 0) + 1}


def make_route_after_synthesis(config: OuroborosConfig):
    def route_after_synthesis(state: OuroborosState) -> str:
        energy = state.get("energy", config.starting_energy)
        depth = state.get("depth", 0)
        guard = state.get("loop_guard", 0)
        mode = state.get("mode", "explore")
        research_done = len(state.get("research_queries", []))

        if energy < 10 or guard > config.max_loop_guard:
            return "surface"
        if depth >= config.max_depth:
            return "surface"
        if depth > 2 and state.get("mood") in ("serene", "melancholic"):
            return "surface"
        if mode in ("analyze", "solve") and research_done == 0 and depth >= 2:
            return "research"
        if random.random() < 0.2 and depth >= 1 and research_done < 2:
            return "research"
        return "think"

    return route_after_synthesis


def should_use_tool(state: OuroborosState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "done"


def make_route_after_breathe(config: OuroborosConfig):
    def route_after_breathe(state: OuroborosState) -> str:
        energy = state.get("energy", config.starting_energy)
        guard = state.get("loop_guard", 0)
        steer_count = state.get("steer_count", 0)
        interval = config.steer_interval

        if energy < 15 or guard > config.max_loop_guard:
            return "__end__"
        if guard > 0 and guard % interval == 0 and steer_count < guard // interval:
            return "steer"
        return "think"

    return route_after_breathe
