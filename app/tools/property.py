"""Mock property lookup tool.

Returns realistic fake property data for testing. The data is deterministic
so that evaluation can compare expected vs actual tool arguments reliably.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Mock property database
# ---------------------------------------------------------------------------

PROPERTIES: dict[str, dict[str, Any]] = {
    "PROP-101": {
        "property_id": "PROP-101",
        "address": "42 Oak Street, Apt 3B",
        "type": "apartment",
        "bedrooms": 3,
        "bathrooms": 2,
        "price": 2500,
        "price_unit": "monthly",
        "available": True,
        "features": ["parking", "gym", "rooftop access"],
        "agent_name": "Sarah Chen",
        "agent_phone": "+1-555-0101",
    },
    "PROP-102": {
        "property_id": "PROP-102",
        "address": "18 Maple Drive",
        "type": "house",
        "bedrooms": 4,
        "bathrooms": 3,
        "price": 450000,
        "price_unit": "sale",
        "available": True,
        "features": ["garden", "garage", "solar panels"],
        "agent_name": "James Rodriguez",
        "agent_phone": "+1-555-0102",
    },
    "PROP-103": {
        "property_id": "PROP-103",
        "address": "7 River Lane, Unit 12",
        "type": "apartment",
        "bedrooms": 1,
        "bathrooms": 1,
        "price": 1400,
        "price_unit": "monthly",
        "available": False,
        "features": ["balcony", "laundry in-unit"],
        "agent_name": "Maria Santos",
        "agent_phone": "+1-555-0103",
    },
    "PROP-104": {
        "property_id": "PROP-104",
        "address": "55 Pine Avenue",
        "type": "townhouse",
        "bedrooms": 2,
        "bathrooms": 2,
        "price": 320000,
        "price_unit": "sale",
        "available": True,
        "features": ["patio", "smart home", "walk-in closet"],
        "agent_name": "Sarah Chen",
        "agent_phone": "+1-555-0101",
    },
}


def property_lookup(property_id: str) -> dict[str, Any]:
    """Look up property details by ID.

    Args:
        property_id: The property identifier (e.g. "PROP-101").

    Returns:
        Property details dict, or an error dict if not found.
    """
    prop = PROPERTIES.get(property_id.upper())
    if prop is None:
        return {
            "error": "property_not_found",
            "message": f"No property found with ID '{property_id}'.",
            "available_ids": list(PROPERTIES.keys()),
        }
    return prop
