import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from car_tools import DataSourceInvalidRequestError, DataSourceUnavailableError
from sqlite_data import SQLiteCarDataSource, initialize_demo_database


class SQLiteCarDataSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "demo.sqlite3"
        initialize_demo_database(self.database_path)
        self.source = SQLiteCarDataSource(self.database_path)

    def test_initializer_does_not_overwrite_existing_database(self):
        with self.assertRaises(FileExistsError):
            initialize_demo_database(self.database_path)
        self.assertEqual(self.source.get_car_profile("42", "1")["model"], "Corsa D")

    def test_profile_is_scoped_to_user_and_vehicle(self):
        self.assertEqual(self.source.get_car_profile("42", "1")["engine"], "Z12XEP")
        self.assertEqual(self.source.get_car_profile("43", "2")["model"], "City")
        with self.assertRaises(DataSourceInvalidRequestError):
            self.source.get_car_profile("42", "2")

    def test_catalog_filters_vehicle_category_and_brand(self):
        both = self.source.search_parts_catalog("1", "front brake pads", None)
        ate = self.source.search_parts_catalog("1", "колодки на переднюю ось", "ate")
        empty = self.source.search_parts_catalog("1", "rear brake discs", None)
        rear = self.source.search_parts_catalog("2", "rear brake discs", None)

        self.assertEqual({part["part_number"] for part in both["parts"]}, {"13.0460-7117.2", "P 59 045"})
        self.assertEqual([part["brand_id"] for part in ate["parts"]], ["brand_ate"])
        self.assertEqual(ate["part_category"], "колодки на переднюю ось")
        self.assertEqual(empty["parts"], [])
        self.assertEqual(rear["parts"][0]["part_number"], "DEMO-RD-2")

    def test_prices_return_only_available_offers(self):
        priced = self.source.get_part_prices(["P 59 045", "DEMO-FP-2"], "RUB")
        missing = self.source.get_part_prices(["DEMO-FP-2"], "RUB")

        self.assertEqual(len(priced["offers"]), 1)
        self.assertEqual(priced["offers"][0]["price"], 4_350)
        self.assertEqual(priced["offers"][0]["seller_id"], "seller_demo_parts_store")
        self.assertTrue(priced["offers"][0]["observed_at"].endswith("Z"))
        self.assertIs(priced["offers"][0]["in_stock"], True)
        self.assertEqual(missing["offers"], [])
        with self.assertRaises(DataSourceInvalidRequestError):
            self.source.get_part_prices(["P 59 045"], "EUR")

    def test_existing_name_based_database_is_read_without_modification(self):
        legacy_path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as connection:
            connection.executescript("""
                CREATE TABLE parts (
                    part_number TEXT PRIMARY KEY, vehicle_id TEXT, category TEXT,
                    brand TEXT, position TEXT
                );
                CREATE TABLE offers (
                    part_number TEXT, seller TEXT, price_rub INTEGER, in_stock INTEGER
                );
                INSERT INTO parts VALUES
                    ('P 59 045', '1', 'front brake pads', 'Brembo', 'front axle');
                INSERT INTO offers VALUES
                    ('P 59 045', 'Demo Parts Store', 4350, 1);
            """)
            connection.commit()
        source = SQLiteCarDataSource(legacy_path)
        catalog = source.search_parts_catalog("1", "front brake pads", "brembo")
        prices = source.get_part_prices(["P 59 045"], "RUB")
        self.assertEqual(catalog["parts"][0]["brand_id"], "brand_brembo")
        self.assertNotIn("brand", catalog["parts"][0])
        self.assertEqual(prices["offers"][0]["seller_id"], "seller_demo_parts_store")
        self.assertIsNone(prices["offers"][0]["observed_at"])
        self.assertNotIn("seller", prices["offers"][0])

    def test_query_values_are_parameters_not_sql(self):
        malicious_brand = "ATE' OR 1=1 --"
        result = self.source.search_parts_catalog("1", "front brake pads", malicious_brand)
        prices = self.source.get_part_prices(["P 59 045') OR 1=1 --"], "RUB")
        self.assertEqual(result["parts"], [])
        self.assertEqual(prices["offers"], [])
        self.assertEqual(self.source.get_car_profile("42", "1")["model"], "Corsa D")

    def test_source_opens_database_read_only(self):
        with self.assertRaises(DataSourceUnavailableError):
            self.source._query("UPDATE vehicles SET model = ? WHERE vehicle_id = ?", ("Changed", "1"))
        self.assertEqual(self.source.get_car_profile("42", "1")["model"], "Corsa D")

    def test_missing_or_corrupt_database_is_unavailable(self):
        with self.assertRaises(DataSourceUnavailableError):
            SQLiteCarDataSource(self.database_path.with_name("missing.sqlite3"))

        corrupt_path = self.database_path.with_name("corrupt.sqlite3")
        corrupt_path.write_text("not a SQLite database", encoding="utf-8")
        with self.assertRaises(DataSourceUnavailableError):
            SQLiteCarDataSource(corrupt_path).get_car_profile("42", "1")


if __name__ == "__main__":
    unittest.main()
