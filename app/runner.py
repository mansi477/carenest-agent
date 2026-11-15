"""Runs one user turn through the graph and reports what happened (nodes, tools, sources)."""
import logging
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app import config as settings
from app.graph import get_graph
from app.utils import message_text

log = logging.getLogger("carenest.runner")

FALLBACK_REPLY = ("Sorry, something went wrong on my side. Please try again, "
                  "or ask for a human agent and I'll connect you.")


@dataclass
class TurnResult:
    reply: str
    route: str
    nodes: list[str] = field(default_factory=list)       # nodes visited this turn, in order
    tool_calls: list[dict] = field(default_factory=list)  # tools the model called this turn
    retrieved: list[dict] = field(default_factory=list)   # KB chunks retrieved (source, heading, score)
    grounded: bool = False
    latency_ms: int = 0
    error: str | None = None


def run_turn(message: str, user_id: str, session_id: str, graph=None) -> TurnResult:
    graph = graph or get_graph()
    cfg = {
        # Namespacing by user_id means one user can never resume another user's session.
        "configurable": {"thread_id": f"{user_id}:{session_id}", "user_id": user_id},
        "recursion_limit": settings.RECURSION_LIMIT,
    }
    started, nodes, error = time.perf_counter(), [], None
    try:
        for update in graph.stream({"messages": [HumanMessage(content=message)]}, cfg, stream_mode="updates"):
            nodes.extend(update.keys())
    except GraphRecursionError:
        error = "recursion_limit"
        log.error("tool loop hit recursion limit")
    except Exception as exc:  # network, auth, provider errors
        error = type(exc).__name__
        log.exception("graph run failed")

    state = graph.get_state(cfg).values or {}
    messages = state.get("messages", [])
    turn_start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    turn = messages[turn_start:]

    tool_calls = [{"name": c["name"], "args": c["args"]}
                  for m in turn if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    reply = next((message_text(m) for m in reversed(turn)
                  if isinstance(m, AIMessage) and message_text(m).strip()), FALLBACK_REPLY)
    if error:
        reply = FALLBACK_REPLY
    retrieved = [{"source": h["source"], "heading": h["heading"], "score": h["score"]}
                 for h in state.get("hits", [])]
    return TurnResult(reply=reply, route=state.get("route", "unknown"), nodes=nodes,
                      tool_calls=tool_calls, retrieved=retrieved, grounded=bool(state.get("grounded")),
                      latency_ms=int((time.perf_counter() - started) * 1000), error=error)
