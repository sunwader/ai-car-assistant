"""Tool implementations and the small contract required of a car data source."""

from __future__ import annotations

from typing import Any, Protocol


JsonObject = dict[str, Any]


class DataSourceUnavailableError(RuntimeError):
    """A data source could not answer; this is not an empty search result."""


class DataSourceInvalidRequestError(ValueError):
    """The source rejected an unsupported request, such as a currency."""


class CarDataSource(Protocol):
    """Read-only source contract.

    Catalog part records return brand_id; price offers return seller_id and
    a source observation timestamp. The application resolves those IDs to
    model-visible display names and validates the timestamp. Unknown part
    numbers produce no offers, never fabricated fallback prices.
    """

    def get_car_profile(self, user_id: str, vehicle_id: str) -> JsonObject: ...

    def search_parts_catalog(
        self, vehicle_id: str, part_category: str, brand: str | None
    ) -> JsonObject: ...

    def get_part_prices(self, part_numbers: list[str], currency: str) -> JsonObject: ...


class CarTools:
    """Keep model-facing tool signatures separate from source implementation."""

    def __init__(self, data_source: CarDataSource) -> None:
        self.data_source = data_source

    def get_car_profile(self, user_id: str, vehicle_id: str) -> JsonObject:
        return self.data_source.get_car_profile(user_id, vehicle_id)

    def search_parts_catalog(
        self, vehicle_id: str, part_category: str, brand: str | None
    ) -> JsonObject:
        return self.data_source.search_parts_catalog(vehicle_id, part_category, brand)

    def get_part_prices(self, part_numbers: list[str], currency: str) -> JsonObject:
        return self.data_source.get_part_prices(part_numbers, currency)
