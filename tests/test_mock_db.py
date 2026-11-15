from app import mock_db as db

USER, OTHER = "u_1001", "u_1002"


def setup_function(_):
    db.reset()


def test_user_sees_own_appointment():
    res = db.get_appointments(USER, "APT-1001")
    assert res["ok"] and res["status"] == "confirmed"
    assert "user_id" not in res


def test_cannot_read_another_users_records():
    assert db.get_appointments(USER, "APT-2001")["ok"] is False
    assert db.get_orders(USER, "ORD-6001")["ok"] is False
    # "not mine" and "does not exist" must look identical
    missing = db.get_appointments(USER, "APT-0000")["error"].replace("APT-0000", "X")
    forbidden = db.get_appointments(USER, "APT-2001")["error"].replace("APT-2001", "X")
    assert missing == forbidden


def test_listing_only_returns_own_records():
    ids = {a["appointment_id"] for a in db.get_appointments(USER)["appointments"]}
    assert ids == {"APT-1001", "APT-1002"}


def test_booking_earliest_slot_removes_it():
    first = db.book_appointment(USER, "dermatology")
    assert first["ok"] and first["status"] == "confirmed"
    second = db.book_appointment(USER, "dermatology")
    assert second["ok"] and second["time"] != first["time"]
    assert db.book_appointment(USER, "dermatology")["ok"] is False  # both slots gone


def test_booking_validates_input():
    assert db.book_appointment(USER, "dermatology", "not-a-date")["ok"] is False
    assert db.book_appointment(USER, "cardiology")["ok"] is False
    assert db.book_appointment(USER, "skin", "", "03:33")["ok"] is False  # alias works, time closed


def test_ticket_ids_increment():
    assert db.create_ticket(USER, "x", "hello") == "TKT-0001"
    assert db.create_ticket(USER, "x", "again") == "TKT-0002"
