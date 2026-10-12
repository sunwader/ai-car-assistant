"""Deterministic demo data for the car tools; no model or API code lives here."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from car_tools import DataSourceInvalidRequestError
from trusted_labels import DEMO_TRUSTED_LABELS


VEHICLE_PROFILES = {
    ("42", "1"): {
        "manufacturer": "Opel",
        "model": "Corsa D",
        "year": 2007,
        "engine": "Z12XEP",
    },
    ("43", "2"): {
        "manufacturer": "Demo Motors",
        "model": "City",
        "year": 2010,
        "engine": "D-100",
    },
}

FRONT_BRAKE_PADS = [
    {
        "brand_id": "brand_ate",
        "part_number": "13.0460-7117.2",
        "position": "front axle",
    },
    {
        "brand_id": "brand_brembo",
        "part_number": "P 59 045",
        "position": "front axle",
    },
]

PARTS_BY_VEHICLE = {
    "1": {"front brake pads": FRONT_BRAKE_PADS},
    "2": {
        "front brake pads": [{
            "brand_id": "brand_demoline",
            "part_number": "DEMO-FP-2",
            "position": "front axle",
        }],
        "rear brake discs": [{
            "brand_id": "brand_demoline",
            "part_number": "DEMO-RD-2",
            "position": "rear axle",
        }],
    },
}

PRICE_OFFERS_RUB = {
    "13.0460-7117.2": (4_900, True),
    "P 59 045": (4_350, True),
    "DEMO-RD-2": (6_200, False),
}


class MockCarDataSource:
    """Read-only demo source with the same operations needed by the tools."""

    def get_car_profile(self, user_id: str, vehicle_id: str) -> dict[str, Any]:
        profile = VEHICLE_PROFILES.get((user_id, vehicle_id))
        if profile is None:
            raise DataSourceInvalidRequestError("Selected vehicle not found for this user")
        return {"user_id": user_id, "vehicle_id": vehicle_id, **profile}

    def search_parts_catalog(
        self, vehicle_id: str, part_category: str, brand: str | None
    ) -> dict[str, Any]:
        normalized_category = part_category.strip().casefold().replace("_", " ")
        is_front_pads = (
            "front" in normalized_category and "pad" in normalized_category
        ) or (
            "перед" in normalized_category and "колод" in normalized_category
        )
        is_rear_discs = (
            "rear" in normalized_category and ("disc" in normalized_category or "rotor" in normalized_category)
        ) or (
            "зад" in normalized_category and "диск" in normalized_category
        )
        category = (
            "front brake pads" if is_front_pads else
            "rear brake discs" if is_rear_discs else normalized_category
        )
        candidates = deepcopy(PARTS_BY_VEHICLE.get(vehicle_id, {}).get(category, []))
        if brand:
            requested_brand_id = DEMO_TRUSTED_LABELS.brand_id_for_name(brand)
            candidates = [
                part for part in candidates if part["brand_id"] == requested_brand_id
            ]

        return {
            "vehicle_id": vehicle_id,
            "part_category": part_category,
            "parts": candidates,
            "note": "Mock compatibility data; verify by VIN before purchase.",
        }

    def get_part_prices(
        self, part_numbers: list[str], currency: str
    ) -> dict[str, Any]:
        if currency != "RUB":
            raise DataSourceInvalidRequestError("Mock prices are available only in RUB")
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        offers = [
            {
                "part_number": part_number,
                "seller_id": "seller_demo_parts_store",
                "price": PRICE_OFFERS_RUB[part_number][0],
                "currency": currency.upper(),
                "in_stock": PRICE_OFFERS_RUB[part_number][1],
                "observed_at": observed_at,
            }
            for part_number in part_numbers
            if part_number in PRICE_OFFERS_RUB
        ]
        return {
            "offers": offers,
            "note": "Mock prices in RUB only; no conversion or purchase has been made.",
        }
