"""Tools the agent can call.

The signed-in user's id comes from the run config, NOT from a model-supplied
argument. The model can never ask for someone else's data by changing a parameter.
"""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app import mock_db


def _user_id(config: RunnableConfig) -> str:
    user_id = (config or {}).get("configurable", {}).get("user_id")
    if not user_id:
        raise PermissionError("No signed-in user for this request.")
    return user_id


@tool
def get_appointment_status(appointment_id: str, config: RunnableConfig) -> dict:
    """Get the status, doctor, date and time of one of the signed-in user's appointments.
    Pass an appointment ID like 'APT-1001'. Pass an empty string to list all of the user's appointments."""
    return mock_db.get_appointments(_user_id(config), appointment_id)


@tool
def get_order_status(order_id: str, config: RunnableConfig) -> dict:
    """Get the status and delivery or report ETA of one of the signed-in user's orders
    (medicine delivery or lab tests). Pass an order ID like 'ORD-5001'.
    Pass an empty string to list all of the user's orders."""
    return mock_db.get_orders(_user_id(config), order_id)


@tool
def find_available_slots(specialty: str) -> dict:
    """List open appointment slots for a specialty, grouped by date (YYYY-MM-DD) with HH:MM times.
    Specialties include 'general physician', 'dermatology' and 'pediatrics'."""
    return mock_db.find_slots(specialty)


@tool
def book_appointment(specialty: str, date: str, time: str, config: RunnableConfig) -> dict:
    """Book an appointment for the signed-in user. This changes data, so only call it after the user
    has confirmed the details. date is YYYY-MM-DD and time is HH:MM (24-hour). Pass an empty string
    for date and/or time to book the earliest available."""
    return mock_db.book_appointment(_user_id(config), specialty, date, time)


TOOLS = [get_appointment_status, get_order_status, find_available_slots, book_appointment]
