"""Offline retrieval eval. No API key needed. Use it to tune chunking, top-k and the score threshold.

    python -m evals.retrieval_eval
    python -m evals.retrieval_eval --top-k 1 --max-chars 300 --min-score 0.2
"""
import argparse
import json
from pathlib import Path

from app import config
from app.rag import Retriever, load_chunks

CASES = Path(__file__).parent / "cases.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    p.add_argument("--max-chars", type=int, default=config.CHUNK_MAX_CHARS)
    p.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = p.parse_args()

    chunks = load_chunks(max_chars=args.max_chars, overlap=args.overlap)
    retriever = Retriever(chunks, top_k=args.top_k, min_score=args.min_score)
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]

    answerable = [c for c in cases if "expected_source" in c]
    refuse = [c for c in cases if c.get("expect_ungrounded")]
    hit_k = hit_1 = 0
    rr_total = 0.0
    misses = []
    for c in answerable:
        query = c.get("retrieval_query") or c["messages"][-1]
        hits = retriever.search(query)
        sources = [h.chunk.source for h in hits]
        if c["expected_source"] in sources:
            hit_k += 1
            rr_total += 1 / (sources.index(c["expected_source"]) + 1)
        else:
            misses.append((c["id"], query, [(h.chunk.heading, round(h.score, 2)) for h in hits]))
        hit_1 += bool(sources) and sources[0] == c["expected_source"]
        if hits and not retriever.is_confident(hits):
            misses.append((c["id"], query, f"below threshold: {hits[0].score:.2f}"))

    refused = 0
    for c in refuse:
        hits = retriever.search(c["messages"][-1])
        if retriever.is_confident(hits):
            misses.append((c["id"], c["messages"][-1], f"answered anyway, top score {hits[0].score:.2f}"))
        else:
            refused += 1

    n = len(answerable)
    print(f"chunks={len(chunks)} top_k={args.top_k} min_score={args.min_score} "
          f"max_chars={args.max_chars} overlap={args.overlap}")
    print(f"hit@{args.top_k}: {hit_k}/{n}   hit@1: {hit_1}/{n}   MRR: {rr_total / n:.2f}")
    print(f"correctly refused (no confident answer): {refused}/{len(refuse)}")
    for case_id, query, info in misses:
        print(f"  MISS {case_id}: {query!r} -> {info}")


if __name__ == "__main__":
    main()
