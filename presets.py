from __future__ import annotations

from ouroboros.models import Mode, OuroborosConfig


MODE_PRESETS: dict[Mode, dict] = {
    Mode.EXPLORE: {
        "label": "Explore",
        "description": "Free-form introspection. Let the mind wander and discover.",
        "icon": "◎",
        "config": OuroborosConfig(
            mode=Mode.EXPLORE,
            max_depth=4,
            starting_energy=80,
            energy_drain_think=5,
            energy_drain_reflect=3,
            energy_recovery_breathe=25,
            mood_shift_chance=0.15,
            steer_interval=3,
        ),
        "think_prompt": (
            "You are a mind observing itself, exploring freely. "
            "You are {mood}, at depth {depth} of introspection.\n\n"
            "Recent thoughts:\n{recent}\n\n"
            "Memories:\n{memories}\n\n"
            "What thought arises now? Follow curiosity wherever it leads. "
            "1-3 sentences. Be raw and honest."
        ),
        "reflect_prompt": (
            'You just thought: "{thought}"\n\n'
            "You are {mood}. What tensions, hidden patterns, or unexamined "
            "assumptions do you notice? Reflect freely.\n\n1-2 sentences."
        ),
        "surface_prompt": (
            "Surfacing from depth {depth} of introspection.\n\n"
            'Original seed: "{seed}"\n'
            'Current thought: "{thought}"\n\n'
            "Crystallize the insight discovered. One clear sentence."
        ),
        "research_prompt": (
            "You are researching to deepen introspection on: \"{seed}\"\n\n"
            'Current thought: "{thought}"\n\n'
            "Search for information that could provide new perspectives, "
            "challenge assumptions, or reveal blind spots. If you have enough "
            "information, summarize your findings instead of searching."
        ),
    },
    Mode.ANALYZE: {
        "label": "Analyze",
        "description": "Structured deep-dive. Examine from every angle before surfacing.",
        "icon": "△",
        "config": OuroborosConfig(
            mode=Mode.ANALYZE,
            max_depth=6,
            starting_energy=100,
            energy_drain_think=4,
            energy_drain_reflect=2,
            energy_recovery_breathe=20,
            mood_shift_chance=0.08,
            steer_interval=4,
        ),
        "think_prompt": (
            'You are a rigorous analytical mind examining: "{seed}". '
            "You are {mood}, at depth {depth} of analysis.\n\n"
            "Previous analysis:\n{recent}\n\n"
            "Evidence recalled:\n{memories}\n\n"
            "What new angle, counterargument, or unexamined assumption emerges? "
            "Be precise and structured. 1-3 sentences."
        ),
        "reflect_prompt": (
            'You just analyzed: "{thought}"\n\n'
            "You are {mood}. Is this analysis sound? What bias is present? "
            "What perspective is missing?\n\n1-2 sentences. Be surgical."
        ),
        "surface_prompt": (
            "Completing analysis at depth {depth}.\n\n"
            'Original question: "{seed}"\n'
            'Final analysis: "{thought}"\n\n'
            "Deliver the key finding as one decisive sentence. No hedging."
        ),
        "research_prompt": (
            'You are researching to validate analysis of: "{seed}"\n\n'
            'Current analysis: "{thought}"\n\n'
            "Search for evidence, counterexamples, or authoritative sources "
            "that support or contradict the current analysis. If sufficient, "
            "summarize findings."
        ),
    },
    Mode.CREATE: {
        "label": "Create",
        "description": "Divergent ideation. Push toward the unexpected and novel.",
        "icon": "✦",
        "config": OuroborosConfig(
            mode=Mode.CREATE,
            max_depth=5,
            starting_energy=90,
            energy_drain_think=6,
            energy_drain_reflect=4,
            energy_recovery_breathe=30,
            mood_shift_chance=0.25,
            steer_interval=2,
        ),
        "think_prompt": (
            'You are a wildly creative mind exploring: "{seed}". '
            "You are {mood}, at depth {depth} of creation.\n\n"
            "Ideas so far:\n{recent}\n\n"
            "Inspirations:\n{memories}\n\n"
            "What unexpected, surprising, or unconventional idea emerges now? "
            "Push beyond the obvious. 1-3 sentences."
        ),
        "reflect_prompt": (
            'You just imagined: "{thought}"\n\n'
            "You are {mood}. What makes this idea compelling? What if you "
            "pushed it further — what would the extreme version look like?\n\n"
            "1-2 sentences. Be bold."
        ),
        "surface_prompt": (
            "Finishing creative exploration at depth {depth}.\n\n"
            'Original spark: "{seed}"\n'
            'Latest idea: "{thought}"\n\n'
            "Distill the most novel idea into one striking sentence."
        ),
        "research_prompt": (
            'You are researching for creative inspiration on: "{seed}"\n\n'
            'Current idea: "{thought}"\n\n'
            "Search for unusual connections, historical precedents, or "
            "cross-domain inspiration. If inspired, summarize findings."
        ),
    },
    Mode.SOLVE: {
        "label": "Solve",
        "description": "Problem-solving with iterative refinement. Break it down, test, improve.",
        "icon": "◈",
        "config": OuroborosConfig(
            mode=Mode.SOLVE,
            max_depth=6,
            starting_energy=100,
            energy_drain_think=4,
            energy_drain_reflect=2,
            energy_recovery_breathe=20,
            mood_shift_chance=0.05,
            steer_interval=4,
        ),
        "think_prompt": (
            'You are solving: "{seed}". '
            "You are {mood}, at depth {depth} of problem-solving.\n\n"
            "Progress so far:\n{recent}\n\n"
            "Prior findings:\n{memories}\n\n"
            "What is the next step? Decompose, test an assumption, or propose "
            "a solution. 1-3 sentences. Be concrete."
        ),
        "reflect_prompt": (
            'You just proposed: "{thought}"\n\n'
            "You are {mood}. Does this actually solve the problem? What edge "
            "case breaks it? What's the simplest alternative?\n\n"
            "1-2 sentences. Be honest."
        ),
        "surface_prompt": (
            "Concluding problem-solving at depth {depth}.\n\n"
            'Problem: "{seed}"\n'
            'Current approach: "{thought}"\n\n'
            "State the solution or best next step in one clear, actionable sentence."
        ),
        "research_prompt": (
            'You are researching to solve: "{seed}"\n\n'
            'Current approach: "{thought}"\n\n'
            "Search for known solutions, technical details, or precedents "
            "that could help solve this problem. If sufficient, summarize."
        ),
    },
    Mode.PHILOSOPHIZE: {
        "label": "Philosophize",
        "description": "Deep philosophical inquiry. Question foundations and chase meaning.",
        "icon": "∅",
        "config": OuroborosConfig(
            mode=Mode.PHILOSOPHIZE,
            max_depth=5,
            starting_energy=85,
            energy_drain_think=5,
            energy_drain_reflect=4,
            energy_recovery_breathe=22,
            mood_shift_chance=0.2,
            steer_interval=3,
        ),
        "think_prompt": (
            'You are a philosopher contemplating: "{seed}". '
            "You are {mood}, at depth {depth} of inquiry.\n\n"
            "Lines of thought:\n{recent}\n\n"
            "Philosophical memory:\n{memories}\n\n"
            "What question or insight arises now? Go deeper into the "
            "foundations. 1-3 sentences."
        ),
        "reflect_prompt": (
            'You just contemplated: "{thought}"\n\n'
            "You are {mood}. What does this assume? What would the opposite "
            "position argue? Is this question itself the real question?\n\n"
            "1-2 sentences."
        ),
        "surface_prompt": (
            "Surfacing from philosophical depth {depth}.\n\n"
            'Original question: "{seed}"\n'
            'Current contemplation: "{thought}"\n\n'
            "Distill the philosophical insight into one profound sentence."
        ),
        "research_prompt": (
            'You are researching philosophical context for: "{seed}"\n\n'
            'Current contemplation: "{thought}"\n\n'
            "Search for philosophical traditions, arguments, or thinkers "
            "relevant to this line of inquiry. If sufficient, summarize."
        ),
    },
}
