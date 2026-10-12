"""Application-owned IDs and display names; source responses supply IDs only."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustedLabels:
    """Immutable ID → name registry curated independently of tool results."""

    brands: tuple[tuple[str, str], ...]
    sellers: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for entries, max_name_length in ((self.brands, 80), (self.sellers, 120)):
            if not isinstance(entries, tuple) or not entries:
                raise ValueError("Trusted label registry cannot be empty")
            ids: set[str] = set()
            names: set[str] = set()
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError("Trusted label entries must be (id, name) pairs")
                identifier, name = entry
                if (
                    not isinstance(identifier, str)
                    or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", identifier)
                    or not isinstance(name, str)
                    or not name.strip()
                    or not name.isprintable()
                    or len(name) > max_name_length
                ):
                    raise ValueError("Trusted label entry is invalid")
                if identifier in ids or name.casefold() in names:
                    raise ValueError("Trusted label IDs and names must be unique")
                ids.add(identifier)
                names.add(name.casefold())

    def brand_name(self, brand_id: str) -> str | None:
        return next((name for identifier, name in self.brands if identifier == brand_id), None)

    def seller_name(self, seller_id: str) -> str | None:
        return next((name for identifier, name in self.sellers if identifier == seller_id), None)

    def brand_id_for_name(self, name: str) -> str | None:
        return next(
            (identifier for identifier, display in self.brands if display.casefold() == name.casefold()),
            None,
        )

    def seller_id_for_name(self, name: str) -> str | None:
        return next(
            (identifier for identifier, display in self.sellers if display.casefold() == name.casefold()),
            None,
        )


DEMO_TRUSTED_LABELS = TrustedLabels(
    brands=(
        ("brand_ate", "ATE"),
        ("brand_brembo", "Brembo"),
        ("brand_demoline", "DemoLine"),
    ),
    sellers=(("seller_demo_parts_store", "Demo Parts Store"),),
)
