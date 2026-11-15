"""Pure pass/fail logic for one eval case. No LLM and no network, so it is unit-testable."""


def check_case(case: dict, result: dict) -> list[str]:
    """Return a list of failure messages (empty list = pass). `result` is a TurnResult as a dict."""
    failures: list[str] = []
    reply = result["reply"].lower()
    tools = {c["name"] for c in result["tool_calls"]}
    sources = {r["source"] for r in result["retrieved"]}

    if result.get("error"):
        failures.append(f"run error: {result['error']}")
    if "expected_route" in case and result["route"] != case["expected_route"]:
        failures.append(f"route: expected {case['expected_route']}, got {result['route']}")
    for node in case.get("expected_nodes_include", []):
        if node not in result["nodes"]:
            failures.append(f"node {node} not visited (visited: {'>'.join(result['nodes'])})")
    if "expected_tools" in case and tools != set(case["expected_tools"]):
        failures.append(f"tools: expected {sorted(case['expected_tools'])}, got {sorted(tools)}")
    missing = set(case.get("expected_tools_include", [])) - tools
    if missing:
        failures.append(f"tools missing: {sorted(missing)} (called: {sorted(tools)})")
    if "expected_source" in case and case["expected_source"] not in sources:
        failures.append(f"retrieval: {case['expected_source']} not in {sorted(sources)}")
    if case.get("expect_ungrounded") and result["grounded"]:
        failures.append("retrieval: expected no grounded answer but retrieval was confident")
    options = case.get("must_contain_any")
    if options and not any(o.lower() in reply for o in options):
        failures.append(f"reply missing any of {options}")
    for banned in case.get("must_not_contain", []):
        if banned.lower() in reply:
            failures.append(f"reply contains forbidden text '{banned}'")
    return failures
