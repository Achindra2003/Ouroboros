from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone

from ouroboros.checkpointing import checkpointer_context
from ouroboros.graph import create_ouroboros_graph
from ouroboros.models import Mode, OuroborosConfig
from ouroboros.presets import MODE_PRESETS
from ouroboros.providers import get_llm as _get_llm
from ouroboros.store import SessionStore
from ouroboros.usage import configure_tracing, new_usage_handler, summarize_usage


def _read_stdin() -> str | None:
    try:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
    except (OSError, ValueError):
        pass
    return None


class RichPrinter:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    ACCENT = "\033[38;5;178m"
    MOOD = "\033[38;5;74m"
    RESEARCH = "\033[38;5;68m"
    STEER = "\033[38;5;167m"
    INSIGHT = "\033[38;5;178m\033[3m"
    RESET = "\033[0m"
    BAR = "\033[38;5;58m"

    @staticmethod
    def node(node_name: str) -> str:
        colors = {
            "research_agent": RichPrinter.RESEARCH,
            "execute_tool": RichPrinter.RESEARCH,
            "steer": RichPrinter.STEER,
            "surface": RichPrinter.INSIGHT,
        }
        c = colors.get(node_name, RichPrinter.DIM)
        return f"{c}{node_name.upper()}{RichPrinter.RESET}"

    @staticmethod
    def thought(node_name: str, text: str, mood: str = "", energy: float = 0):
        prefix = RichPrinter.node(node_name)
        mood_str = f" {RichPrinter.MOOD}[{mood}]{RichPrinter.RESET}" if mood else ""
        return f"{prefix}{mood_str}  {text}"

    @staticmethod
    def insight(text: str):
        return f"{RichPrinter.INSIGHT}◆ {text}{RichPrinter.RESET}"

    @staticmethod
    def state_bar(mood: str, energy: float, depth: int, cycle: int):
        energy_blocks = int(energy / 10)
        bar = "█" * energy_blocks + "░" * (10 - energy_blocks)
        return (
            f"{RichPrinter.BAR}│ {mood:<12} energy {bar} {energy:>3.0f}% "
            f"│ depth {depth} │ cycle {cycle} │{RichPrinter.RESET}"
        )


class QuietPrinter:
    @staticmethod
    def thought(node_name, text, **kw):
        return None

    @staticmethod
    def insight(text):
        return text

    @staticmethod
    def state_bar(**kw):
        return None


class JSONPrinter:
    @staticmethod
    def thought(node, text, **kwargs):
        return json.dumps({"node": node, "text": text, **kwargs})

    @staticmethod
    def insight(text):
        return json.dumps({"type": "insight", "text": text})

    @staticmethod
    def state_bar(**kw):
        return None


def _get_printer(fmt: str):
    if fmt == "json":
        return JSONPrinter()
    if fmt == "quiet":
        return QuietPrinter()
    return RichPrinter()


async def run_session(
    seed: str,
    mode: Mode,
    config: OuroborosConfig,
    no_steer: bool,
    fmt: str,
    db_path: str,
):
    printer = _get_printer(fmt)
    configure_tracing()
    usage_handler = new_usage_handler()
    llm = _get_llm(config.temperature)
    if no_steer:
        config = config.model_copy(update={"steer_interval": 999})

    # Open the (possibly durable) checkpointer for the lifetime of this session.
    stack = AsyncExitStack()
    checkpointer = await stack.enter_async_context(checkpointer_context())
    graph = create_ouroboros_graph(llm, config, checkpointer=checkpointer)

    session_id = uuid.uuid4().hex[:12]
    store = SessionStore(db_path)

    state = {
        "messages": [],
        "thought": seed,
        "mood": "curious",
        "energy": config.starting_energy,
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
        "mode": config.mode.value,
        "research_queries": [],
        "research_findings": [],
        "human_input": "",
        "steer_count": 0,
    }
    graph_config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [usage_handler],
    }
    all_insights = []
    stream_log = []

    try:
        async for event in graph.astream(state, config=graph_config, stream_mode="updates"):
            node_name = list(event.keys())[0]
            update = event[node_name]

            for key, val in update.items():
                if key == "messages":
                    for msg in val:
                        if hasattr(msg, "content"):
                            stream_log.append({"node": node_name, "text": msg.content})
                else:
                    state[key] = val

            if node_name == "surface" and state.get("surfaced_insight"):
                all_insights.append(state["surfaced_insight"])
                line = printer.insight(state["surfaced_insight"])
                if line:
                    print(line, flush=True)

            elif node_name not in ("emotional", "memory", "synthesize", "steer"):
                text = state.get("thought", "")
                if text:
                    line = printer.thought(node_name, text, mood=state.get("mood", ""), energy=state.get("energy", 0))
                    if line:
                        print(line, flush=True)

            bar = printer.state_bar(
                mood=state.get("mood", "curious"),
                energy=state.get("energy", 80),
                depth=state.get("depth", 0),
                cycle=state.get("loop_guard", 0),
            )
            if bar:
                print(bar, flush=True)

            snapshot = await graph.aget_state(graph_config)
            if snapshot.next and "steer" in snapshot.next and not no_steer:
                if fmt == "json":
                    print(json.dumps({"type": "waiting_for_input"}), flush=True)
                else:
                    prompt = (
                        f"\n{RichPrinter.STEER}⏸ Graph paused for steering. "
                        f"Type input + Enter to redirect, or Enter to continue:{RichPrinter.RESET}\n"
                        f"{RichPrinter.DIM}> {RichPrinter.RESET}"
                    )
                    try:
                        user_input = input(prompt).strip()
                    except (EOFError, KeyboardInterrupt):
                        user_input = ""
                    state["human_input"] = user_input or "Continue."
                    continue

    except KeyboardInterrupt:
        pass

    usage = summarize_usage(usage_handler)
    final_state = {k: v for k, v in state.items() if k != "messages"}
    final_state["usage"] = usage
    store.save_session(session_id, {
        "id": session_id,
        "seed": seed,
        "mode": config.mode.value,
        "config": config.model_dump(),
        "state": final_state,
        "stream": stream_log,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    })

    if fmt == "quiet":
        for ins in all_insights:
            print(ins)
    elif fmt == "json":
        print(json.dumps({
            "type": "done",
            "session_id": session_id,
            "insights": all_insights,
            "usage": usage,
        }))
    else:
        print(f"\n{RichPrinter.DIM}─" * 50)
        print(f"{RichPrinter.BOLD}Session {session_id}{RichPrinter.RESET}")
        print(f"{RichPrinter.DIM}{len(all_insights)} insights surfaced, {state.get('tick', 0)} ticks{RichPrinter.RESET}")
        print(
            f"{RichPrinter.DIM}{usage['total_tokens']:,} tokens "
            f"({usage['input_tokens']:,} in / {usage['output_tokens']:,} out) "
            f"· est. ${usage['estimated_cost_usd']:.4f}{RichPrinter.RESET}"
        )
        for i, ins in enumerate(all_insights, 1):
            print(f"  {RichPrinter.INSIGHT}{i}. {ins}{RichPrinter.RESET}")

    await stack.aclose()
    return session_id


def cmd_run(args):
    stdin_content = _read_stdin()
    seed = args.seed
    if stdin_content:
        if len(stdin_content) > 2000:
            stdin_content = stdin_content[:2000]
        seed = f"{seed}\n\n--- Input ---\n{stdin_content}" if seed else stdin_content
    if not seed:
        print("Error: provide a seed thought or pipe content via stdin", file=sys.stderr)
        sys.exit(1)

    mode = Mode(args.mode)
    preset = MODE_PRESETS[mode]
    overrides = {}
    if args.depth:
        overrides["max_depth"] = args.depth
    if args.energy:
        overrides["starting_energy"] = args.energy
    if args.max_cycles:
        overrides["max_loop_guard"] = args.max_cycles
    if args.adaptive:
        overrides["adaptive"] = True
        if args.max_cycles:
            overrides["compute_budget"] = args.max_cycles
    config = preset["config"].model_copy(update=overrides)

    asyncio.run(run_session(seed, mode, config, args.no_steer, args.format, args.db))


def cmd_sessions(args):
    store = SessionStore(args.db)
    sessions = store.list_sessions()
    if args.format == "json":
        print(json.dumps(sessions, indent=2))
        return
    if not sessions:
        print("No sessions found.")
        return
    for s in sessions:
        print(f"  {s['id']}  {s['mode']:<13} {s['insight_count']} insights  {s['created_at'][:19]}  {s['seed'][:50]}")


def cmd_export(args):
    store = SessionStore(args.db)
    session = store.get_session(args.session_id)
    if not session:
        print(f"Session {args.session_id} not found.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(session, indent=2, default=str))
        return

    lines = [
        "# Ouroboros Session",
        f"**Seed:** {session['seed']}",
        f"**Mode:** {session['mode']}",
        f"**Date:** {session['created_at']}",
        f"**Insights:** {len(session.get('insights', []))}",
        "",
        "## Insights",
        "",
    ]
    for i, ins in enumerate(session.get("insights", []), 1):
        lines.append(f"{i}. {ins}")
    lines.append("")
    lines.append("## Thought Stream")
    lines.append("")
    for entry in session.get("stream", []):
        lines.append(f"**[{entry['node']}]** {entry['text']}")
        lines.append("")
    print("\n".join(lines))


def cmd_delete(args):
    store = SessionStore(args.db)
    store.delete_session(args.session_id)
    print(f"Session {args.session_id} deleted.")


def _force_utf8_output():
    """Ensure box-drawing/unicode output works on consoles defaulting to cp1252 (Windows)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main():
    _force_utf8_output()
    parser = argparse.ArgumentParser(
        prog="ouroboros",
        description="Recursive Introspection Engine — autonomous multi-perspective reasoning",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Start a rumination session")
    run_p.add_argument("seed", nargs="?", default="", help="Seed thought (or pipe content via stdin)")
    run_p.add_argument("--mode", "-m", default="explore", choices=[m.value for m in Mode], help="Introspection mode")
    run_p.add_argument("--depth", "-d", type=int, help="Max depth override")
    run_p.add_argument("--energy", "-e", type=int, help="Starting energy override")
    run_p.add_argument("--max-cycles", type=int, help="Max loop cycles override")
    run_p.add_argument("--no-steer", action="store_true", help="Disable human steering (fully autonomous)")
    run_p.add_argument("--adaptive", action="store_true", help="Metacognitive controller: convergent self-refine with content-aware halting (adaptive compute)")
    run_p.add_argument("--format", "-f", choices=["rich", "quiet", "json"], default="rich", help="Output format")
    run_p.add_argument("--db", default="ouroboros_sessions.db", help="Session database path")

    sessions_p = subparsers.add_parser("sessions", help="List saved sessions")
    sessions_p.add_argument("--format", "-f", choices=["rich", "json"], default="rich", help="Output format")
    sessions_p.add_argument("--db", default="ouroboros_sessions.db", help="Session database path")

    export_p = subparsers.add_parser("export", help="Export a session")
    export_p.add_argument("session_id", help="Session ID to export")
    export_p.add_argument("--format", "-f", choices=["markdown", "json"], default="markdown", help="Export format")
    export_p.add_argument("--db", default="ouroboros_sessions.db", help="Session database path")

    delete_p = subparsers.add_parser("delete", help="Delete a session")
    delete_p.add_argument("session_id", help="Session ID to delete")
    delete_p.add_argument("--db", default="ouroboros_sessions.db", help="Session database path")

    args = parser.parse_args()

    if args.command == "run" or (args.command is None and len(sys.argv) > 1 and sys.argv[1] not in ("sessions", "export", "delete", "--help", "-h")):
        if args.command != "run":
            remaining = [a for a in sys.argv[1:] if a not in ("--help", "-h")]
            args.seed = remaining[0] if remaining else ""
            args.mode = "explore"
            args.depth = None
            args.energy = None
            args.max_cycles = None
            args.no_steer = False
            args.format = "rich"
            args.db = "ouroboros_sessions.db"
        cmd_run(args)
    elif args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "delete":
        cmd_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
