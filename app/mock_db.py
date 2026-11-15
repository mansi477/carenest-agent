"""In-memory fake backend (appointments, orders, slots, tickets).

Stands in for the internal systems a real assistant would reach through
APIs or MCP servers. Every read is scoped to a user_id so one customer can
never see another customer's records.
"""
from copy import deepcopy
from datetime import date, timedelta

USERS = {
    "u_1001": {"name": "Asha Rao"},
    "u_1002": {"name": "Rahul Mehta"},
}

SPECIALTY_ALIASES = {
    "gp": "general physician",
    "physician": "general physician",
    "general": "general physician",
    "general medicine": "general physician",
    "skin": "dermatology",
    "dermatologist": "dermatology",
    "paediatrics": "pediatrics",
    "pediatrician": "pediatrics",
    "child specialist": "pediatrics",
}


def _day(offset: int) -> str:
    return (date.today() + timedelta(days=offset)).isoformat()


def _seed() -> dict:
    return {
        "appointments": {
            "APT-1001": {"user_id": "u_1001", "specialty": "general physician",
                         "doctor": "Dr. Meera Iyer", "date": _day(2), "time": "10:30",
                         "mode": "video", "status": "confirmed"},
            "APT-1002": {"user_id": "u_1001", "specialty": "dermatology",
                         "doctor": "Dr. Sanjay Kulkarni", "date": _day(-5), "time": "15:00",
                         "mode": "in-clinic", "status": "completed"},
            "APT-2001": {"user_id": "u_1002", "specialty": "pediatrics",
                         "doctor": "Dr. Latha Nair", "date": _day(1), "time": "17:00",
                         "mode": "video", "status": "cancelled"},
        },
        "orders": {
            "ORD-5001": {"user_id": "u_1001", "type": "medicine delivery",
                         "items": ["Paracetamol 650mg x 10", "ORS sachets x 5"],
                         "status": "out_for_delivery", "eta": f"{_day(0)} by 8 PM"},
            "ORD-5002": {"user_id": "u_1001", "type": "diagnostics",
                         "items": ["Complete Blood Count"],
                         "status": "sample_collected", "report_eta": f"{_day(1)} by noon"},
            "ORD-6001": {"user_id": "u_1002", "type": "medicine delivery",
                         "items": ["Vitamin D3 60000 IU x 4"],
                         "status": "delivered", "eta": _day(-2)},
        },
        "slots": {
            "general physician": {_day(1): ["09:00", "11:30"], _day(2): ["10:30", "16:00"]},
            "dermatology": {_day(3): ["14:00", "15:00"]},
            "pediatrics": {_day(1): ["17:00"], _day(4): ["10:00", "10:30"]},
        },
        "tickets": [],
    }


_state = _seed()


def reset() -> None:
    """Restore seed data (used by tests and evals)."""
    global _state
    _state = deepcopy(_seed())


def normalize_specialty(raw: str) -> str:
    key = (raw or "").strip().lower()
    return SPECIALTY_ALIASES.get(key, key)


def get_appointments(user_id: str, appointment_id: str = "") -> dict:
    appts = _state["appointments"]
    if appointment_id:
        record = appts.get(appointment_id.strip().upper())
        # Same message for "missing" and "belongs to someone else": don't leak existence.
        if not record or record["user_id"] != user_id:
            return {"ok": False, "error": f"No appointment {appointment_id} found on your account."}
        return {"ok": True, "appointment_id": appointment_id.strip().upper(), **_public(record)}
    mine = [{"appointment_id": k, **_public(v)} for k, v in appts.items() if v["user_id"] == user_id]
    return {"ok": True, "appointments": sorted(mine, key=lambda a: (a["date"], a["time"]))}


def get_orders(user_id: str, order_id: str = "") -> dict:
    orders = _state["orders"]
    if order_id:
        record = orders.get(order_id.strip().upper())
        if not record or record["user_id"] != user_id:
            return {"ok": False, "error": f"No order {order_id} found on your account."}
        return {"ok": True, "order_id": order_id.strip().upper(), **_public(record)}
    mine = [{"order_id": k, **_public(v)} for k, v in orders.items() if v["user_id"] == user_id]
    return {"ok": True, "orders": mine}


def find_slots(specialty: str) -> dict:
    spec = normalize_specialty(specialty)
    days = _state["slots"].get(spec)
    available = {d: t for d, t in sorted((days or {}).items()) if t}
    if not available:
        known = sorted(_state["slots"])
        return {"ok": False, "error": f"No open slots for '{specialty}'.", "specialties_with_slots": known}
    return {"ok": True, "specialty": spec, "slots": available}


def book_appointment(user_id: str, specialty: str, date_str: str = "", time_str: str = "") -> dict:
    """Book a slot. Empty date means earliest available date; empty time means earliest time."""
    found = find_slots(specialty)
    if not found["ok"]:
        return found
    spec, slots = found["specialty"], found["slots"]

    if date_str:
        try:
            date.fromisoformat(date_str)
        except ValueError:
            return {"ok": False, "error": f"Invalid date '{date_str}'. Use YYYY-MM-DD."}
        if date_str not in slots:
            return {"ok": False, "error": f"No {spec} slots on {date_str}.", "available_slots": slots}
        chosen_date = date_str
    else:
        chosen_date = next(iter(slots))

    times = slots[chosen_date]
    if time_str and time_str not in times:
        return {"ok": False, "error": f"{time_str} is not open on {chosen_date}.",
                "available_slots": {chosen_date: times}}
    chosen_time = time_str or times[0]

    _state["slots"][spec][chosen_date].remove(chosen_time)
    new_id = f"APT-{9000 + len(_state['appointments'])}"
    _state["appointments"][new_id] = {
        "user_id": user_id, "specialty": spec, "doctor": "To be assigned",
        "date": chosen_date, "time": chosen_time, "mode": "video", "status": "confirmed",
    }
    return {"ok": True, "appointment_id": new_id, **_public(_state["appointments"][new_id])}


def create_ticket(user_id: str, reason: str, summary: str) -> str:
    ticket_id = f"TKT-{len(_state['tickets']) + 1:04d}"
    _state["tickets"].append({"id": ticket_id, "user_id": user_id, "reason": reason, "summary": summary})
    return ticket_id


def tickets() -> list:
    return list(_state["tickets"])


def _public(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "user_id"}
