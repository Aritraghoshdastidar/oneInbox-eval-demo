"""Mock calendar / scheduling tools.

Provides check_availability and book_appointment with deterministic behaviour
so evaluators can verify tool calls reliably.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.tools.property import PROPERTIES

# ---------------------------------------------------------------------------
# Mock schedule data — predefined availability
# ---------------------------------------------------------------------------

# Keys: (property_id, date, time) → available?
_AVAILABILITY: dict[tuple[str, str, str], bool] = {
    ("PROP-101", "2026-09-20", "10:00"): True,
    ("PROP-101", "2026-09-20", "14:00"): True,
    ("PROP-101", "2026-09-20", "16:00"): False,
    ("PROP-101", "2026-09-21", "10:00"): True,
    ("PROP-102", "2026-09-20", "10:00"): True,
    ("PROP-102", "2026-09-20", "14:00"): False,
    ("PROP-102", "2026-09-21", "11:00"): True,
    ("PROP-103", "2026-09-20", "10:00"): False,
    ("PROP-104", "2026-09-20", "10:00"): True,
    ("PROP-104", "2026-09-20", "15:00"): True,
}

# Track booked appointments (mutable state for the mock)
_BOOKINGS: list[dict[str, Any]] = []


def check_availability(
    property_id: str, date: str, time: str
) -> dict[str, Any]:
    """Check if a property viewing slot is available.

    Args:
        property_id: Property ID (e.g. "PROP-101").
        date: Date string in YYYY-MM-DD format.
        time: Time string in HH:MM format.

    Returns:
        Availability result dict, or error dict if the property does not
        exist in the catalog.
    """
    # INVARIANT: property must exist in the catalog.
    # Without this guard, calling check_availability for an unknown property
    # (e.g. PROP-999) would silently return available=True, masking an
    # incorrect tool-routing decision.  This was identified as a critical
    # environment bug in V4B where the model called check_availability
    # instead of property_lookup for PROP-999 and received a fake
    # "available" result instead of NOT_FOUND.
    if property_id.upper() not in PROPERTIES:
        return {
            "error": "property_not_found",
            "message": f"No property found with ID '{property_id}'.",
            "available_ids": list(PROPERTIES.keys()),
        }

    key = (property_id.upper(), date, time)
    if key in _AVAILABILITY:
        available = _AVAILABILITY[key]
    else:
        # Default: any slot not explicitly listed is available
        available = True

    result: dict[str, Any] = {
        "property_id": property_id.upper(),
        "date": date,
        "time": time,
        "available": available,
    }

    if not available:
        # Suggest alternative slots
        alternatives = [
            {"date": d, "time": t}
            for (pid, d, t), avail in _AVAILABILITY.items()
            if pid == property_id.upper() and avail
        ]
        result["suggested_alternatives"] = alternatives[:3]

    return result


def book_appointment(
    property_id: str, date: str, time: str,
    contact_name: str | None = None,
    contact_phone: str | None = None,
    contact_email: str | None = None,
) -> dict[str, Any]:
    """Book a property viewing appointment.

    Args:
        property_id: Property ID.
        date: Date string in YYYY-MM-DD format.
        time: Time string in HH:MM format.
        contact_name: Name of the person booking.
        contact_phone: Phone number (optional).
        contact_email: Email address (optional).

    Returns:
        Booking confirmation or error dict.
    """
    # INVARIANT: property must exist in the catalog before booking.
    if property_id.upper() not in PROPERTIES:
        return {
            "success": False,
            "error": "property_not_found",
            "message": f"No property found with ID '{property_id}'.",
            "available_ids": list(PROPERTIES.keys()),
        }

    # Check availability first
    avail = check_availability(property_id, date, time)
    if not avail["available"]:
        return {
            "success": False,
            "error": "slot_unavailable",
            "message": f"The slot on {date} at {time} for {property_id} is not available.",
            "suggested_alternatives": avail.get("suggested_alternatives", []),
        }

    booking = {
        "booking_id": f"BK-{uuid.uuid4().hex[:8].upper()}",
        "property_id": property_id.upper(),
        "date": date,
        "time": time,
        "contact_name": contact_name,
        "contact_phone": contact_phone,
        "contact_email": contact_email,
        "status": "confirmed",
    }
    _BOOKINGS.append(booking)

    return {
        "success": True,
        "booking": booking,
        "message": f"Viewing confirmed for {property_id} on {date} at {time}.",
    }


def reset_bookings() -> None:
    """Reset booking state between test runs."""
    _BOOKINGS.clear()
