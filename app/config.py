"""Central settings. Everything tunable lives here so evals can compare runs."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"

# --- LLM ---------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
_DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))

# --- Retrieval ---------------------------------------------------------
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
# Below this cosine score we treat the knowledge base as having no answer.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))

# --- Agent -------------------------------------------------------------
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))  # user turns of history sent to the LLM
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "12"))  # hard stop for tool loops
