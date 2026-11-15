import pytest

pytest.importorskip("langchain_core")  # tool wrappers need langchain; skip if not installed

from app import mock_db  # noqa: E402
from app.tools import book_appointment, get_appointment_status  # noqa: E402


def _cfg(user_id):
    return {"configurable": {"user_id": user_id}}


def setup_function(_):
    mock_db.reset()


def test_tool_uses_user_from_config_not_from_model():
    own = get_appointment_status.invoke({"appointment_id": "APT-1001"}, config=_cfg("u_1001"))
    other = get_appointment_status.invoke({"appointment_id": "APT-1001"}, config=_cfg("u_1002"))
    assert own["ok"] is True
    assert other["ok"] is False


def test_book_appointment_through_tool():
    res = book_appointment.invoke({"specialty": "dermatology", "date": "", "time": ""}, config=_cfg("u_1001"))
    assert res["ok"] is True


def test_tool_without_user_is_rejected():
    with pytest.raises(PermissionError):
        get_appointment_status.invoke({"appointment_id": ""}, config={"configurable": {}})
