"""One place that decides which chat model the app talks to."""
from functools import lru_cache

from langchain.chat_models import init_chat_model

from app import config


@lru_cache(maxsize=1)
def get_llm():
    # temperature 0: we want repeatable behaviour so eval differences come from our changes, not sampling.
    return init_chat_model(config.LLM_MODEL, model_provider=config.LLM_PROVIDER, temperature=0)
