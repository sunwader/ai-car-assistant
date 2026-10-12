"""Reusable conformance checks for the read-only CarDataSource boundary.

A future adapter can subclass SourceContractChecks with unittest.TestCase,
provide a fresh source in setUp(), and choose fixture IDs/part numbers. These
checks exercise the app's own result validator, not a duplicate schema.
"""

import tempfile
import unittest
from pathlib import Path

from agent_loop import RunState
from car_tools import DataSourceInvalidRequestError
from mock_data import MockCarDataSource
from sqlite_data import SQLiteCarDataSource, initialize_demo_database
from trusted_labels import DEMO_TRUSTED_LABELS


class SourceContractChecks:
    user_id = "42"
    vehicle_id = "1"
    category = "front brake pads"
    brand = "Brembo"
    brand_id = "brand_brembo"
    part_number = "P 59 045"
    unknown_part_number = "NO-SUCH-PART"
    other_user_id = "43"
    other_vehicle_id = "2"
    other_part_number = "DEMO-FP-2"
    trusted_labels = DEMO_TRUSTED_LABELS

    def checked_state(self):
        state = RunState(
            user_id=self.user_id, vehicle_id=self.vehicle_id,
            trusted_labels=self.trusted_labels,
        )
        profile_args = {"user_id": self.user_id, "vehicle_id": self.vehicle_id}
        state.validate("get_car_profile", profile_args)
        state.record("get_car_profile", self.source.get_car_profile(**profile_args), profile_args)
        return state

    def test_profile_and_filtered_catalog_pass_application_contract(self):
        state = self.checked_state()
        catalog_args = {
            "vehicle_id": self.vehicle_id,
            "part_category": self.category,
            "brand": self.brand,
        }
        state.validate("search_parts_catalog", catalog_args)
        catalog = self.source.search_parts_catalog(**catalog_args)
        state.record("search_parts_catalog", catalog, catalog_args)

        self.assertIn(self.part_number, state.catalog_part_numbers)
        self.assertTrue(all(part["brand_id"] == self.brand_id for part in catalog["parts"]))

    def test_observed_offer_passes_application_contract(self):
        state = self.checked_state()
        catalog_args = {
            "vehicle_id": self.vehicle_id,
            "part_category": self.category,
            "brand": self.brand,
        }
        state.validate("search_parts_catalog", catalog_args)
        catalog = self.source.search_parts_catalog(**catalog_args)
        state.record("search_parts_catalog", catalog, catalog_args)

        price_args = {"part_numbers": [self.part_number], "currency": "RUB"}
        state.validate("get_part_prices", price_args)
        prices = self.source.get_part_prices(**price_args)
        state.record("get_part_prices", prices, price_args)

        self.assertTrue(prices["offers"])
        self.assertTrue(all(offer["part_number"] == self.part_number for offer in prices["offers"]))
        self.assertTrue(state.price_checked)

    def test_unknown_part_has_no_fabricated_offer(self):
        result = self.source.get_part_prices([self.unknown_part_number], "RUB")
        self.assertEqual(result["offers"], [])

        mixed = self.source.get_part_prices(
            [self.part_number, self.unknown_part_number], "RUB"
        )
        self.assertTrue(mixed["offers"])
        self.assertTrue(all(
            offer["part_number"] == self.part_number for offer in mixed["offers"]
        ))

    def test_profile_requires_matching_user_and_vehicle(self):
        with self.assertRaises(DataSourceInvalidRequestError):
            self.source.get_car_profile(self.user_id, self.other_vehicle_id)
        with self.assertRaises(DataSourceInvalidRequestError):
            self.source.get_car_profile(self.other_user_id, self.vehicle_id)

        other = self.source.get_car_profile(self.other_user_id, self.other_vehicle_id)
        self.assertEqual(other["vehicle_id"], self.other_vehicle_id)
        self.assertEqual(other["user_id"], self.other_user_id)

    def test_catalog_filters_by_vehicle_not_just_category(self):
        current = self.source.search_parts_catalog(self.vehicle_id, self.category, None)
        other = self.source.search_parts_catalog(self.other_vehicle_id, self.category, None)
        self.assertIn(self.part_number, {part["part_number"] for part in current["parts"]})
        self.assertNotIn(self.part_number, {part["part_number"] for part in other["parts"]})
        self.assertIn(self.other_part_number, {part["part_number"] for part in other["parts"]})

    def test_unsupported_currency_is_distinct_from_no_offers(self):
        with self.assertRaises(DataSourceInvalidRequestError):
            self.source.get_part_prices([self.part_number], "EUR")


class MockSourceContractTests(SourceContractChecks, unittest.TestCase):
    def setUp(self):
        self.source = MockCarDataSource()


class SQLiteSourceContractTests(SourceContractChecks, unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "contract.sqlite3"
        initialize_demo_database(path)
        self.source = SQLiteCarDataSource(path)


if __name__ == "__main__":
    unittest.main()
