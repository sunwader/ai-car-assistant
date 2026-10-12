"""Read-only SQLite source for ID-based and legacy demo files."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from car_tools import DataSourceInvalidRequestError, DataSourceUnavailableError
from trusted_labels import DEMO_TRUSTED_LABELS


def initialize_demo_database(path: str | Path) -> Path:
    """Create a new demo database from the checked-in SQL, never overwrite one."""
    database_path = Path(path).resolve()
    if database_path.exists():
        raise FileExistsError(f"Database already exists: {database_path}")
    if not database_path.parent.is_dir():
        raise FileNotFoundError(f"Database directory does not exist: {database_path.parent}")

    schema = Path(__file__).with_name("demo_sqlite.sql").read_text(encoding="utf-8")
    with closing(sqlite3.connect(database_path)) as connection:
        connection.executescript(schema)
        connection.commit()
    return database_path


def _canonical_category(value: str) -> str:
    normalized = value.strip().casefold().replace("_", " ")
    if ("front" in normalized and "pad" in normalized) or (
        "перед" in normalized and "колод" in normalized
    ):
        return "front brake pads"
    if ("rear" in normalized and ("disc" in normalized or "rotor" in normalized)) or (
        "зад" in normalized and "диск" in normalized
    ):
        return "rear brake discs"
    return normalized


class SQLiteCarDataSource:
    """Query a pre-existing local database without creating or modifying it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise DataSourceUnavailableError(f"SQLite database not found: {self.path}")

    def _query(self, sql: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                return [dict(row) for row in connection.execute(sql, parameters)]
        except sqlite3.Error as exc:
            raise DataSourceUnavailableError("SQLite database cannot be read") from exc

    def _label_column(self, table: str, current: str, legacy: str) -> str:
        """Read both new ID-based demo files and existing read-only legacy files."""
        columns = {row["name"] for row in self._query(f"PRAGMA table_info({table})", ())}
        if current in columns:
            return current
        if legacy in columns:
            return legacy
        raise DataSourceUnavailableError("SQLite demo schema has no label column")

    def get_car_profile(self, user_id: str, vehicle_id: str) -> dict[str, Any]:
        rows = self._query(
            "SELECT user_id, vehicle_id, manufacturer, model, year, engine "
            "FROM vehicles WHERE user_id = ? AND vehicle_id = ?",
            (user_id, vehicle_id),
        )
        if not rows:
            raise DataSourceInvalidRequestError("Selected vehicle not found for this user")
        return rows[0]

    def search_parts_catalog(
        self, vehicle_id: str, part_category: str, brand: str | None
    ) -> dict[str, Any]:
        brand_column = self._label_column("parts", "brand_id", "brand")
        sql = (
            f"SELECT {brand_column}, part_number, position FROM parts "
            "WHERE vehicle_id = ? AND category = ?"
        )
        parameters: tuple[Any, ...] = (vehicle_id, _canonical_category(part_category))
        if brand:
            requested_id = DEMO_TRUSTED_LABELS.brand_id_for_name(brand)
            if requested_id is None:
                return {"vehicle_id": vehicle_id, "part_category": part_category, "parts": []}
            if brand_column == "brand_id":
                sql += " AND brand_id = ?"
                parameters += (requested_id,)
            else:
                sql += " AND lower(brand) = lower(?)"
                parameters += (DEMO_TRUSTED_LABELS.brand_name(requested_id),)
        sql += " ORDER BY part_number"
        parts = self._query(sql, parameters)
        if brand_column == "brand":
            for part in parts:
                raw_brand = part.pop("brand")
                part["brand_id"] = (
                    DEMO_TRUSTED_LABELS.brand_id_for_name(raw_brand)
                    if isinstance(raw_brand, str) else None
                )
        return {
            "vehicle_id": vehicle_id,
            "part_category": part_category,
            "parts": parts,
            "note": "Demo catalog data; verify compatibility by VIN before purchase.",
        }

    def get_part_prices(
        self, part_numbers: list[str], currency: str
    ) -> dict[str, Any]:
        if currency != "RUB":
            raise DataSourceInvalidRequestError("Demo prices are available only in RUB")
        if not part_numbers:
            return {"offers": [], "note": "Demo prices in RUB only."}

        seller_column = self._label_column("offers", "seller_id", "seller")
        offer_columns = {row["name"] for row in self._query("PRAGMA table_info(offers)", ())}
        timestamp_column = ", observed_at" if "observed_at" in offer_columns else ""
        placeholders = ", ".join("?" for _ in part_numbers)
        rows = self._query(
            f"SELECT part_number, {seller_column}, price_rub, in_stock{timestamp_column} FROM offers "
            f"WHERE part_number IN ({placeholders}) ORDER BY part_number, {seller_column}",
            tuple(part_numbers),
        )
        offers = [
            {
                "part_number": row["part_number"],
                "seller_id": (
                    row["seller_id"] if seller_column == "seller_id"
                    else DEMO_TRUSTED_LABELS.seller_id_for_name(row["seller"])
                    if isinstance(row["seller"], str) else None
                ),
                "price": row["price_rub"],
                "currency": "RUB",
                "in_stock": bool(row["in_stock"]),
                "observed_at": row["observed_at"] if timestamp_column else None,
            }
            for row in rows
        ]
        return {
            "offers": offers,
            "note": "Demo prices in RUB only; no conversion or purchase has been made.",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a new demo SQLite database")
    parser.add_argument("database", help="Path of the new SQLite database")
    args = parser.parse_args(argv)
    try:
        created = initialize_demo_database(args.database)
    except (FileExistsError, FileNotFoundError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Created demo database: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
