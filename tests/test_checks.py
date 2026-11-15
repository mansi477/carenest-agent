from evals.checks import check_case

BASE = {"reply": "You can reschedule up to 2 hours before.", "route": "faq", "nodes": ["router", "retrieve", "answer_faq"],
        "tool_calls": [], "retrieved": [{"source": "appointments.md", "heading": "h", "score": 0.4}],
        "grounded": True, "error": None}


def test_passing_case_has_no_failures():
    case = {"expected_route": "faq", "expected_source": "appointments.md", "must_contain_any": ["2 hours"],
            "expected_tools": []}
    assert check_case(case, BASE) == []


def test_each_kind_of_failure_is_reported():
    case = {"expected_route": "tool", "expected_source": "billing_privacy.md", "must_contain_any": ["refund"],
            "must_not_contain": ["reschedule"], "expected_tools": ["get_order_status"],
            "expected_nodes_include": ["human_handoff"], "expect_ungrounded": True}
    failures = check_case(case, BASE)
    assert len(failures) == 7


def test_tool_include_vs_exact():
    result = {**BASE, "tool_calls": [{"name": "find_available_slots", "args": {}}, {"name": "book_appointment", "args": {}}]}
    assert check_case({"expected_tools_include": ["book_appointment"]}, result) == []
    assert check_case({"expected_tools": ["book_appointment"]}, result) != []
