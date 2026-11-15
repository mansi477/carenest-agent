"""Knowledge-base retrieval: heading-aware chunking + TF-IDF search.

TF-IDF is a deliberate baseline: no model download, no API key, fully
inspectable. Its weakness is vocabulary mismatch ("money back" vs "refund").
When you hit that failure, swap Retriever for an embedding-based one; the
rest of the app only depends on `search()` and `is_confident()`.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import config


# Words that appear everywhere in this knowledge base and say nothing about the topic.
_DOMAIN_STOP_WORDS = {"carenest", "app", "tap", "need", "want"}
_SUFFIXES = ("ations", "ation", "ating", "ated", "ings", "ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'reschedule' matches 'rescheduling'. Crude on purpose."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words
            if w not in ENGLISH_STOP_WORDS and w not in _DOMAIN_STOP_WORDS]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str   # file name, e.g. "appointments.md"
    heading: str  # section heading the chunk came from
    text: str     # heading + body, so heading words are searchable


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        # Plain dict so it can live in LangGraph state and be checkpointed.
        return {"id": self.chunk.id, "source": self.chunk.source,
                "heading": self.chunk.heading, "text": self.chunk.text,
                "score": round(self.score, 4)}


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split on '## ' headings. Returns [(heading, body), ...]."""
    sections, heading, lines = [], "Overview", []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif line.startswith("# "):
            continue  # document title
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _window(body: str, max_chars: int, overlap: int) -> list[str]:
    """Pack whole sentences into pieces of <= max_chars, carrying `overlap` chars forward."""
    if len(body) <= max_chars:
        return [body]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
    pieces, cur = [], []
    for sentence in sentences:
        if cur and len(" ".join(cur)) + 1 + len(sentence) > max_chars:
            pieces.append(" ".join(cur))
            carry, size = [], 0
            for prev in reversed(cur):
                if size + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                size += len(prev)
            cur = carry
        cur.append(sentence)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def load_chunks(kb_dir: Path | None = None, max_chars: int | None = None,
                overlap: int | None = None) -> list[Chunk]:
    kb_dir = kb_dir or config.KB_DIR
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        for heading, body in _split_sections(path.read_text(encoding="utf-8")):
            for n, piece in enumerate(_window(body, max_chars, overlap)):
                chunks.append(Chunk(id=f"{path.stem}:{heading}:{n}", source=path.name,
                                    heading=heading, text=f"{heading}. {piece}"))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk], top_k: int | None = None, min_score: float | None = None):
        if not chunks:
            raise ValueError("Knowledge base is empty. Check the kb/ folder.")
        self.chunks = chunks
        self.top_k = top_k or config.TOP_K
        self.min_score = config.MIN_SCORE if min_score is None else min_score
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                           ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self.top_k
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def is_confident(self, hits: list[Hit]) -> bool:
        return bool(hits) and hits[0].score >= self.min_score


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_chunks())
