from __future__ import annotations

import random
import re

from langchain_core.messages import AIMessage
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

FALLBACK_RESEARCH = [
    "The search turns up nothing the mind did not already hold. Inward, then.",
    "No external answer arrives. The thought must do its own work.",
    "Research falls silent; the question returns, unaltered, to the thinker.",
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


def derive_mood(thought: str, current_mood: str, config: OuroborosConfig) -> str:
    """Derive the next mood from the thought's content.

    Keyword detection acts as a cheap, deterministic prior; absent a match, the
    mood may stochastically shift. Routing and the breathe node depend on these
    transitions, so this stays a pure, fast function (no LLM).
    """
    thought_lower = thought.lower()
    for candidate, words in MOOD_KEYWORDS.items():
        if any(w in thought_lower for w in words):
            return candidate
    if random.random() < config.mood_shift_chance:
        return random.choice(MOOD_SHIFTS.get(current_mood, ["curious"]))
    return current_mood


def make_emotional_analysis(llm: BaseChatModel, config: OuroborosConfig):
    async def emotional_analysis(state: OuroborosState) -> dict:
        thought = state.get("thought", "")
        new_mood = derive_mood(thought, state.get("mood", "curious"), config)
        prompt = (
            f'A mind in a "{new_mood}" state is examining this thought:\n'
            f'"{thought}"\n\n'
            "Speak from the emotional/affective perspective ONLY (not logic). "
            "What feeling underlies this thought? What is it avoiding, or yearning "
            "toward? One or two sentences."
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            reading = resp.content.strip()
        except Exception:
            reading = f"Emotional undertone: {new_mood}. {MOOD_READINGS.get(new_mood, '')}"
        return {"mood": new_mood, "emotional_reading": reading}

    return emotional_analysis


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
    from ouroboros.memory import semantic_search

    thought = state.get("thought", "")
    mems = state.get("memories", [])
    if not mems:
        return {"memory_reading": "No connected memories. The thought is unanchored."}
    results = semantic_search(thought, mems, k=3)
    related = [m for m, score in results if score > 0] or mems[-2:]
    reading = "Connected memories: " + "; ".join(related)
    return {"memory_reading": reading}


_CONFIDENCE_RE = re.compile(r"confidence\s*[:=]\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE)


def _parse_confidence(text: str) -> tuple[str, float]:
    """Split a trailing ``CONFIDENCE: <0-1>`` marker off an answer.

    Returns ``(answer_without_marker, confidence)``; defaults to 0.5 when the
    model omits or mangles the marker.
    """
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return text.strip(), 0.5
    conf = max(0.0, min(1.0, float(match.group(1))))
    answer = _CONFIDENCE_RE.sub("", text).strip().rstrip("\n").strip()
    return answer, conf


def make_synthesize(llm: BaseChatModel, config: OuroborosConfig | None = None):
    adaptive = bool(config and config.adaptive)

    async def synthesize(state: OuroborosState) -> dict:
        emo = state.get("emotional_reading", "")
        logic = state.get("logical_reading", "")
        mem = state.get("memory_reading", "")
        thought = state.get("thought", "")

        if adaptive:
            return await _synthesize_adaptive(llm, config, state, emo, logic, mem)

        prompt = (
            f'Three perspectives have examined the thought:\n"{thought}"\n\n'
            f"- Emotional: {emo}\n"
            f"- Logical: {logic}\n"
            f"- Memory: {mem}\n\n"
            "Integrate these perspectives into one synthesis. Where do they agree or "
            "conflict? Which tension matters most, and what should the mind examine "
            "next? 2-3 sentences."
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            synthesis = resp.content.strip()
        except Exception:
            synthesis = " ".join(p for p in (emo, logic, mem) if p).strip()
        return {
            "synthesis": synthesis,
            "depth": state.get("depth", 0) + 1,
        }

    return synthesize


async def _synthesize_adaptive(
    llm: BaseChatModel,
    config: OuroborosConfig,
    state: OuroborosState,
    emo: str,
    logic: str,
    mem: str,
) -> dict:
    """Convergent self-refine: improve the *current best answer to the seed*,
    anchored so it cannot drift, then measure convergence and let the controller
    decide whether to halt. Fixes the divergence/drift that made the legacy loop
    lose to single-shot, and produces the metacognitive signals."""
    from ouroboros.graph.controller import answer_stability, decide

    seed = state.get("seed", "")
    prev = state.get("synthesis", "")
    prompt = (
        f'You are refining an answer to this exact question:\n"{seed}"\n\n'
        f"Current best answer:\n{prev or '(none yet — write the first answer)'}\n\n"
        "Three perspectives just examined the latest thinking:\n"
        f"- Emotional: {emo}\n- Logical: {logic}\n- Memory: {mem}\n\n"
        "Critique the current answer in light of these, then write an IMPROVED, "
        "complete, self-contained answer to the question above. Stay strictly on "
        "that question; be concrete and non-obvious; 2-4 sentences. Do not describe "
        "your process. Finally, on a new line, rate how settled your answer now is "
        'as "CONFIDENCE: <0.0-1.0>".'
    )
    try:
        resp = await llm.ainvoke([{"role": "system", "content": prompt}])
        answer, confidence = _parse_confidence(resp.content.strip())
    except Exception:
        answer = prev or " ".join(p for p in (emo, logic, mem) if p).strip()
        confidence = 0.5

    stability = answer_stability(prev, answer)
    new_depth = state.get("depth", 0) + 1
    decision = decide(
        depth=new_depth, stability=stability, confidence=confidence, config=config
    )
    return {
        "synthesis": answer,
        "prev_synthesis": prev,
        "confidence": confidence,
        "stability": stability,
        "should_halt": decision.halt,
        "stop_reason": decision.reason,
        "depth": new_depth,
        # Feed the refined answer forward so the next cycle's think/critique is
        # anchored to it (not to a drifting free-association).
        "messages": [AIMessage(content=answer)],
    }


def make_surface(llm: BaseChatModel, config: OuroborosConfig):
    prompt_template = _get_prompt(config.mode, "surface_prompt")
    adaptive = bool(config and config.adaptive)

    async def surface(state: OuroborosState) -> dict:
        # Adaptive mode: the converged synthesis *is* the answer. Surface it as-is
        # (full, complete, anchored) rather than re-generating a compressed
        # aphorism — this is the output-parity fix for a fair comparison.
        if adaptive:
            answer = state.get("synthesis", "") or state.get("thought", "")
            return {
                "surfaced_insight": answer,
                "messages": [AIMessage(content=f"[answer] {answer}")],
                "insights": [answer],
            }

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


def make_plan_research(llm: BaseChatModel, config: OuroborosConfig):
    """Plan research: ask the LLM for a few focused sub-queries to investigate.

    Produces ``pending_queries`` which ``fan_out_research`` turns into one parallel
    worker per query via the LangGraph ``Send`` API (dynamic map-reduce).
    """

    async def plan_research(state: OuroborosState) -> dict:
        seed = state.get("seed", "")
        thought = state.get("thought", "")
        prompt = (
            f'You are planning research to deepen introspection on: "{seed}"\n'
            f'Current thought: "{thought}"\n\n'
            "List 2-3 focused web-search queries that would surface new perspectives "
            "or challenge an assumption. One query per line, no numbering or commentary."
        )
        try:
            resp = await llm.ainvoke([{"role": "system", "content": prompt}])
            lines = [ln.strip("-•*0123456789. \t") for ln in resp.content.splitlines()]
            queries = [ln for ln in lines if ln][:3]
        except Exception:
            queries = []
        if not queries:
            queries = [thought or seed or "introspection"]
        return {
            "pending_queries": queries,
            "research_queries": queries,
            "tick": state.get("tick", 0) + 1,
        }

    return plan_research


def make_research_worker():
    """One parallel research worker per sub-query (fanned out via Send)."""
    from ouroboros.graph.tools import web_search

    async def research_worker(state: dict) -> dict:
        query = (state or {}).get("query", "")
        try:
            result = web_search.invoke({"query": query})
        except Exception:
            result = random.choice(FALLBACK_RESEARCH)
        return {"research_findings": [f"{query} → {result}"[:300]]}

    return research_worker


def fan_out_research(state: OuroborosState):
    """Dynamic fan-out: emit one Send per pending query, or skip to think."""
    from langgraph.types import Send

    queries = state.get("pending_queries", []) or []
    if not queries:
        return "think"
    return [Send("research_worker", {"query": q}) for q in queries]


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
        # Adaptive mode: the metacognitive controller already decided in synthesize
        # (stored on state). Honour it; allow one research detour for grounded modes
        # before convergence.
        if config.adaptive:
            if state.get("should_halt"):
                return "surface"
            mode = state.get("mode", "explore")
            research_done = len(state.get("research_queries", []))
            if mode in ("analyze", "solve") and research_done == 0 and state.get("depth", 0) >= 1:
                return "research"
            return "think"

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


def make_route_after_breathe(config: OuroborosConfig):
    def route_after_breathe(state: OuroborosState) -> str:
        # Adaptive mode produces one converged answer per run; the compute budget
        # is enforced inside the refine loop, so end once we've surfaced.
        if config.adaptive:
            return "__end__"

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
