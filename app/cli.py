"""Terminal chat for quick manual testing:  python -m app.cli --user u_1001"""
import argparse
import uuid

from app.runner import run_turn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="u_1001", help="u_1001 (Asha) or u_1002 (Rahul)")
    args = parser.parse_args()
    session = uuid.uuid4().hex[:8]
    print(f"CareNest assistant (user {args.user}). Type 'quit' to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        r = run_turn(text, args.user, session)
        print(f"bot> {r.reply}")
        extras = f"route={r.route} nodes={'>'.join(r.nodes)} {r.latency_ms}ms"
        if r.tool_calls:
            extras += " tools=" + ",".join(c["name"] for c in r.tool_calls)
        print(f"     [{extras}]\n")


if __name__ == "__main__":
    main()
