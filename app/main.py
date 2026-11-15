"""FastAPI service.  Run:  uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import mock_db
from app.graph import get_graph
from app.runner import run_turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_graph()  # build at startup so a missing API key or bad config fails immediately
    yield


app = FastAPI(title="CareNest Support Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    # In production user_id comes from a verified auth token, never from the request body.
    user_id: str = Field(examples=["u_1001"])
    session_id: str = Field(min_length=1, max_length=64, examples=["session-1"])
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    route: str
    nodes: list[str]
    tool_calls: list[dict]
    retrieved: list[dict]
    grounded: bool
    latency_ms: int
    error: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.user_id not in mock_db.USERS:
        raise HTTPException(status_code=401, detail="Unknown user")
    # Sync endpoint: FastAPI runs it in a worker thread, which suits the blocking LLM calls.
    return ChatResponse(**asdict(run_turn(req.message, req.user_id, req.session_id)))
