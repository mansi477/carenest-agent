"""End-to-end evals through the real graph. Needs an LLM API key.

    python -m evals.run_evals
    python -m evals.run_evals --category tool
    python -m evals.run_evals --id faq_fasting
    python -m evals.run_evals --baseline evals/results/<earlier-run>.json   # show regressions

Each run is saved to evals/results/ so you can answer "did my change help?" with numbers.
"""
import argparse
import json
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app import config, mock_db
from app.runner import run_turn
from evals.checks import check_case

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category")
    p.add_argument("--id")
    p.add_argument("--baseline", help="path to an earlier results file to compare against")
    args = p.parse_args()

    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]

    rows = []
    for case in cases:
        mock_db.reset()  # booking in one case must not change the next
        session, result = uuid.uuid4().hex, None
        for message in case["messages"]:  # multi-turn cases: checks apply to the final turn
            result = run_turn(message, case.get("user_id", "u_1001"), session)
        failures = check_case(case, asdict(result))
        rows.append({"id": case["id"], "category": case["category"], "passed": not failures,
                     "failures": failures, "reply": result.reply, "route": result.route,
                     "nodes": result.nodes, "tools": [t["name"] for t in result.tool_calls],
                     "latency_ms": result.latency_ms})
        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']:<28} {result.latency_ms:>5} ms")
        for f in failures:
            print(f"        - {f}")

    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += r["passed"]
        by_cat[r["category"]][1] += 1
    print("\nBy category:")
    for cat, (ok, total) in sorted(by_cat.items()):
        print(f"  {cat:<10} {ok}/{total}")
    passed = sum(r["passed"] for r in rows)
    avg = sum(r["latency_ms"] for r in rows) // max(len(rows), 1)
    print(f"Overall: {passed}/{len(rows)} passed, avg latency {avg} ms")

    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"), "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL, "top_k": config.TOP_K, "min_score": config.MIN_SCORE,
            "chunk_max_chars": config.CHUNK_MAX_CHARS, "chunk_overlap": config.CHUNK_OVERLAP}
    out = HERE / "results" / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"Saved {out}")

    if args.baseline:
        old = {r["id"]: r["passed"] for r in json.loads(Path(args.baseline).read_text())["rows"]}
        regressions = [r["id"] for r in rows if old.get(r["id"]) and not r["passed"]]
        fixes = [r["id"] for r in rows if old.get(r["id"]) is False and r["passed"]]
        print(f"\nVs baseline: {len(fixes)} fixed {fixes}, {len(regressions)} regressed {regressions}")


if __name__ == "__main__":
    main()
