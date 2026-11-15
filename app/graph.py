r"""The LangGraph agent.

    START -> router -+-> retrieve -> answer_faq            (grounded)       -> END
                     |            \-> human_handoff          (nothing found)  -> END
                     +-> tool_agent <-> tools                                -> END
                     +-> human_handoff | emergency | medical_advice | smalltalk -> END

State is small and explicit. Each node reads what it needs and returns only what it changes.
"""
import logging
from datetime import date
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app import config as settings
from app import mock_db
from app.llm import get_llm
from app.prompts import (EMERGENCY_REPLY, FAQ_PROMPT, HANDOFF_REPLY, MEDICAL_ADVICE_REPLY,
                         ROUTER_PROMPT, SMALLTALK_PROMPT, TOOL_AGENT_PROMPT)
from app.rag import get_retriever
from app.tools import TOOLS
from app.utils import last_turns, latest_user_text, plain_history, render_transcript

log = logging.getLogger("carenest.graph")


# --------------------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # full conversation (checkpointed per session)
    route: str               # router's decision for the current turn
    standalone_query: str    # latest message rewritten to be self-contained
    hits: list[dict]         # retrieved KB chunks for the current turn
    grounded: bool           # did retrieval find something good enough to answer from?
    handoff_reason: str      # why we escalated, for the ticket


class RouteDecision(BaseModel):
    route: Literal["faq", "tool", "human", "emergency", "medical_advice", "smalltalk"] = Field(
        description="Where to send the user's latest message.")
    standalone_query: str = Field(
        description="The latest message rewritten as a self-contained question.")


@lru_cache(maxsize=1)
def _router_llm():
    return get_llm().with_structured_output(RouteDecision)


@lru_cache(maxsize=1)
def _tool_llm():
    return get_llm().bind_tools(TOOLS)


# --------------------------------------------------------------------------- nodes
def router_node(state: AgentState) -> dict:
    latest = latest_user_text(state["messages"])
    transcript = render_transcript(last_turns(state["messages"], settings.HISTORY_TURNS))
    fresh_turn = {"hits": [], "grounded": False, "handoff_reason": ""}  # clear last turn's leftovers
    try:
        decision = _router_llm().invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=f"Conversation so far:\n{transcript}\n\nClassify the latest user message."),
        ])
    except Exception:
        log.exception("router call failed")
        decision = None
    if decision is None:  # fail safe: a human can always handle it
        return {**fresh_turn, "route": "human", "standalone_query": latest,
                "handoff_reason": "router_failure"}
    query = decision.standalone_query.strip() or latest
    log.info("route=%s query=%r", decision.route, query)
    return {**fresh_turn, "route": decision.route, "standalone_query": query}


def retrieve_node(state: AgentState) -> dict:
    retriever = get_retriever()
    hits = retriever.search(state["standalone_query"])
    grounded = retriever.is_confident(hits)
    log.info("retrieved=%s grounded=%s", [(h.chunk.heading, round(h.score, 3)) for h in hits], grounded)
    update: dict = {"hits": [h.as_dict() for h in hits], "grounded": grounded}
    if not grounded:
        update["handoff_reason"] = "no_grounded_answer"
    return update


def answer_faq_node(state: AgentState) -> dict:
    context = "\n\n".join(f"[{h['source']} > {h['heading']}]\n{h['text']}" for h in state["hits"])
    reply = get_llm().invoke([
        SystemMessage(content=FAQ_PROMPT.format(context=context)),
        HumanMessage(content=state["standalone_query"]),
    ])
    return {"messages": [reply]}


def tool_agent_node(state: AgentState) -> dict:
    system = SystemMessage(content=TOOL_AGENT_PROMPT.format(today=date.today().isoformat()))
    history = last_turns(state["messages"], settings.HISTORY_TURNS)
    return {"messages": [_tool_llm().invoke([system, *history])]}


def human_handoff_node(state: AgentState, config: RunnableConfig) -> dict:
    user_id = config["configurable"]["user_id"]
    reason = state.get("handoff_reason") or "user_request_or_unsupported_action"
    ticket_id = mock_db.create_ticket(user_id, reason, latest_user_text(state["messages"]))
    log.info("handoff ticket=%s reason=%s", ticket_id, reason)
    return {"messages": [AIMessage(content=HANDOFF_REPLY.format(ticket_id=ticket_id))]}


def emergency_node(state: AgentState) -> dict:
    # Deliberately no LLM call: safety messages must be deterministic.
    return {"messages": [AIMessage(content=EMERGENCY_REPLY)]}


def medical_advice_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=MEDICAL_ADVICE_REPLY)]}


def smalltalk_node(state: AgentState) -> dict:
    history = plain_history(state["messages"], settings.HISTORY_TURNS)
    return {"messages": [get_llm().invoke([SystemMessage(content=SMALLTALK_PROMPT), *history])]}


# --------------------------------------------------------------------------- edges
def route_after_router(state: AgentState) -> str:
    return state["route"]


def route_after_retrieve(state: AgentState) -> str:
    return "answer_faq" if state["grounded"] else "human_handoff"


def route_after_tool_agent(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


# --------------------------------------------------------------------------- build
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer_faq", answer_faq_node)
    g.add_node("tool_agent", tool_agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("human_handoff", human_handoff_node)
    g.add_node("emergency", emergency_node)
    g.add_node("medical_advice", medical_advice_node)
    g.add_node("smalltalk", smalltalk_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {
        "faq": "retrieve", "tool": "tool_agent", "human": "human_handoff",
        "emergency": "emergency", "medical_advice": "medical_advice", "smalltalk": "smalltalk"})
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"answer_faq": "answer_faq", "human_handoff": "human_handoff"})
    g.add_conditional_edges("tool_agent", route_after_tool_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "tool_agent")
    for node in ("answer_faq", "human_handoff", "emergency", "medical_advice", "smalltalk"):
        g.add_edge(node, END)

    # MemorySaver keeps sessions in process memory. Use a Postgres/Redis checkpointer in production.
    return g.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
