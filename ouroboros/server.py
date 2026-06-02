from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ouroboros.config import get_settings
from ouroboros.graph import create_ouroboros_graph
from ouroboros.models import Mode
from ouroboros.presets import MODE_PRESETS
from ouroboros.providers import get_llm as _get_llm
from ouroboros.store import SessionStore
from ouroboros.usage import configure_tracing, new_usage_handler, summarize_usage

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
configure_tracing()
settings = get_settings()

app = FastAPI(title="Ouroboros", version="2.1.0", description="Recursive Introspection Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

store = SessionStore()
_ws_clients: set[WebSocket] = set()
_sessions: dict[str, dict] = {}
_graph_tasks: dict[str, asyncio.Task] = {}
_graphs: dict[str, object] = {}
_running: set[str] = set()
_waiting_for_input: set[str] = set()
_usage_handlers: dict[str, object] = {}


def _serialize_step(session_id: str, node: str, session: dict) -> dict:
    s = session["state"]
    return {
        "type": "step",
        "session_id": session_id,
        "node": node,
        "thought": s.get("thought", ""),
        "mood": s.get("mood", "curious"),
        "energy": s.get("energy", 80),
        "depth": s.get("depth", 0),
        "memories": s.get("memories", []),
        "insights": s.get("insights", []),
        "loop_guard": s.get("loop_guard", 0),
        "tick": s.get("tick", 0),
        "surfaced_insight": s.get("surfaced_insight", ""),
        "emotional_reading": s.get("emotional_reading", ""),
        "logical_reading": s.get("logical_reading", ""),
        "memory_reading": s.get("memory_reading", ""),
        "synthesis": s.get("synthesis", ""),
        "mode": session.get("mode", "explore"),
        "stream": session["stream"][-40:],
        "research_queries": s.get("research_queries", []),
        "research_findings": s.get("research_findings", []),
        "steer_count": s.get("steer_count", 0),
        "running": True,
    }


async def _broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _stream_graph(session_id: str, session: dict, graph, graph_config: dict, inputs):
    """Drive the graph, streaming LLM tokens live and per-node state updates.

    Uses LangGraph multi-mode streaming: ``messages`` yields token chunks (so the
    UI can type thoughts out as the model produces them) and ``updates`` yields
    the post-node state delta (so the graph node lights up and readings refresh).
    """
    try:
        async for mode, data in graph.astream(
            inputs, config=graph_config, stream_mode=["updates", "messages"]
        ):
            if session_id not in _running:
                break

            if mode == "messages":
                chunk, meta = data
                text = getattr(chunk, "content", "") or ""
                if text:
                    await _broadcast({
                        "type": "token",
                        "session_id": session_id,
                        "node": meta.get("langgraph_node", ""),
                        "text": text,
                    })
                continue

            # mode == "updates": {node_name: state_delta}
            node_name = next(iter(data))
            update = data[node_name]
            for key, val in update.items():
                if key == "messages":
                    for msg in val:
                        if hasattr(msg, "content"):
                            session["stream"].append({"node": node_name, "text": msg.content})
                else:
                    session["state"][key] = val
            try:
                await _broadcast(_serialize_step(session_id, node_name, session))
            except Exception:
                pass
            # Small beat so the node-by-node graph animation stays watchable.
            await asyncio.sleep(0.15)

        state_snapshot = await graph.aget_state(graph_config)
        if state_snapshot.next and "steer" in state_snapshot.next:
            _waiting_for_input.add(session_id)
            await _broadcast({"type": "waiting_for_input", "session_id": session_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        await _broadcast({"type": "error", "session_id": session_id, "message": str(e)})


async def _run_graph(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        return
    _running.add(session_id)
    _waiting_for_input.discard(session_id)

    config = session["config"]
    llm = _get_llm(temperature=config.temperature)
    graph = create_ouroboros_graph(llm, config)
    _graphs[session_id] = graph

    session["state"] = {
        "messages": [],
        "thought": session["seed"],
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
        "seed": session["seed"],
        "surfaced_insight": "",
        "mode": config.mode.value,
        "research_queries": [],
        "research_findings": [],
        "human_input": "",
        "steer_count": 0,
    }
    session["stream"] = []
    usage_handler = new_usage_handler()
    _usage_handlers[session_id] = usage_handler
    graph_config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [usage_handler],
    }

    await _stream_graph(session_id, session, graph, graph_config, session["state"])

    if session_id in _running:
        _running.discard(session_id)
    usage = summarize_usage(_usage_handlers.get(session_id) or new_usage_handler())
    session["state"]["usage"] = usage
    session["ended_at"] = datetime.now(timezone.utc).isoformat()
    store.save_session(session_id, {
        **{k: v for k, v in session.items() if k != "usage_handler"},
        "mode": config.mode.value,
        "config": config.model_dump(),
    })
    await _broadcast({
        "type": "complete", "session_id": session_id, "running": False, "usage": usage,
    })


async def _resume_graph(session_id: str, human_input: str):
    if session_id not in _waiting_for_input:
        return
    _waiting_for_input.discard(session_id)
    _running.add(session_id)

    session = _sessions.get(session_id)
    graph = _graphs.get(session_id)
    if not session or not graph:
        return

    usage_handler = _usage_handlers.get(session_id) or new_usage_handler()
    _usage_handlers[session_id] = usage_handler
    graph_config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [usage_handler],
    }
    session["state"]["human_input"] = human_input

    await _stream_graph(session_id, session, graph, graph_config, None)

    if session_id in _running:
        _running.discard(session_id)
    usage = summarize_usage(_usage_handlers.get(session_id) or new_usage_handler())
    session["state"]["usage"] = usage
    session["ended_at"] = datetime.now(timezone.utc).isoformat()
    store.save_session(session_id, {
        **{k: v for k, v in session.items() if k != "usage_handler"},
        "mode": session["config"].mode.value,
        "config": session["config"].model_dump(),
    })
    await _broadcast({
        "type": "complete", "session_id": session_id, "running": False, "usage": usage,
    })


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Ouroboros</h1>")


@app.get("/api/modes")
async def get_modes():
    return [
        {
            "mode": m.value,
            "label": p["label"],
            "description": p["description"],
            "icon": p["icon"],
            "config": p["config"].model_dump(),
        }
        for m, p in MODE_PRESETS.items()
    ]


@app.post("/api/start")
async def start_run(seed: str = "What am I?", mode: str = "explore", config: dict = None):
    if len(_running) >= settings.max_concurrent_sessions:
        return JSONResponse(
            status_code=429,
            content={"error": "server busy: too many concurrent sessions, try again shortly"},
        )

    mode_enum = Mode(mode) if mode in [m.value for m in Mode] else Mode.EXPLORE
    preset = MODE_PRESETS[mode_enum]
    overrides = dict(config or {})
    if settings.demo_mode:
        # Clamp untrusted public input to keep free-tier usage bounded.
        requested = overrides.get("max_loop_guard", preset["config"].max_loop_guard)
        overrides["max_loop_guard"] = min(requested, settings.max_demo_cycles)
    cfg = preset["config"].model_copy(update=overrides)

    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = {
        "id": session_id,
        "seed": seed,
        "config": cfg,
        "state": {},
        "stream": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
    }

    task = asyncio.create_task(_run_graph(session_id))
    _graph_tasks[session_id] = task
    return {"status": "started", "session_id": session_id}


@app.post("/api/steer/{session_id}")
async def steer_run(session_id: str, human_input: str = ""):
    if session_id not in _waiting_for_input:
        return {"status": "not_waiting"}
    if not human_input:
        human_input = "Continue exploring."
    task = asyncio.create_task(_resume_graph(session_id, human_input))
    _graph_tasks[f"{session_id}_steer"] = task
    return {"status": "resumed"}


@app.post("/api/stop/{session_id}")
async def stop_run(session_id: str):
    _running.discard(session_id)
    _waiting_for_input.discard(session_id)
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset_run():
    for sid in list(_running):
        _running.discard(sid)
    _waiting_for_input.clear()
    return {"status": "reset"}


@app.get("/api/sessions")
async def list_sessions():
    db_sessions = store.list_sessions()
    live = []
    for sid, s in _sessions.items():
        if not any(d["id"] == sid for d in db_sessions):
            insights = s["state"].get("insights", [])
            live.append(
                {
                    "id": sid,
                    "seed": s["seed"],
                    "mode": s.get("mode", s["config"].mode.value),
                    "created_at": s["created_at"],
                    "insight_count": len(insights),
                    "total_ticks": s["state"].get("tick", 0),
                    "live": True,
                }
            )
    return db_sessions + live


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        db_session = store.get_session(session_id)
        if not db_session:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return db_session
    insights = session["state"].get("insights", [])
    return {
        "id": session["id"],
        "seed": session["seed"],
        "mode": session["config"].mode.value,
        "config": session["config"].model_dump(),
        "state": {k: v for k, v in session["state"].items() if k != "messages"},
        "stream": session["stream"][-80:],
        "insights": insights,
        "created_at": session["created_at"],
        "ended_at": session.get("ended_at"),
        "waiting_for_input": session_id in _waiting_for_input,
    }


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json"):
    session = _sessions.get(session_id)
    if not session:
        session = store.get_session(session_id)
        if not session:
            return JSONResponse(status_code=404, content={"error": "not found"})
    insights = session.get("state", {}).get("insights", []) if isinstance(session, dict) and "state" in session else session.get("insights", [])

    if format == "markdown":
        lines = [
            "# Ouroboros Session",
            f"**Seed:** {session.get('seed', '')}",
            f"**Mode:** {session.get('mode', '')}",
            f"**Date:** {session.get('created_at', '')}",
            "",
            "## Insights",
            "",
        ]
        for i, ins in enumerate(insights, 1):
            lines.append(f"{i}. {ins}")
        lines.append("")
        lines.append("## Thought Stream")
        lines.append("")
        for entry in session.get("stream", []):
            lines.append(f"**[{entry['node']}]** {entry['text']}")
            lines.append("")
        return JSONResponse(content={"markdown": "\n".join(lines)})

    return {
        "meta": {
            "id": session.get("id", ""),
            "seed": session.get("seed", ""),
            "mode": session.get("mode", ""),
            "created_at": session.get("created_at", ""),
            "insight_count": len(insights),
        },
        "thoughts": session.get("stream", []),
        "insights": insights,
        "final_state": session.get("state", {}),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    store.delete_session(session_id)
    return {"status": "deleted"}


@app.get("/api/state")
async def get_state():
    active = next((sid for sid in _running if sid in _sessions), None)
    if active and active in _sessions:
        return _serialize_step(active, "idle", _sessions[active]) | {
            "running": True,
            "waiting_for_input": active in _waiting_for_input,
        }
    return {"running": False, "stream": []}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        active = next((sid for sid in _running if sid in _sessions), None)
        if active:
            await ws.send_text(
                json.dumps(
                    _serialize_step(active, "idle", _sessions[active])
                    | {"running": True, "waiting_for_input": active in _waiting_for_input}
                )
            )
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "start":
                seed = msg.get("seed", "What am I?")
                mode = msg.get("mode", "explore")
                config_overrides = msg.get("config", {})
                resp = await start_run(seed, mode, config_overrides)
                await ws.send_text(json.dumps({"type": "start_ack", **resp}))
            elif msg.get("action") == "steer":
                sid = msg.get("session_id", "")
                human_input = msg.get("input", "Continue.")
                resp = await steer_run(sid, human_input)
                await ws.send_text(json.dumps({"type": "steer_ack", **resp}))
            elif msg.get("action") == "stop":
                sid = msg.get("session_id", "")
                await stop_run(sid)
                await ws.send_text(json.dumps({"type": "stop_ack"}))
            elif msg.get("action") == "reset":
                await reset_run()
                await ws.send_text(json.dumps({"type": "reset_ack"}))
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
