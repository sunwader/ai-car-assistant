"""Application-owned label provenance tests."""

import unittest

from trusted_labels import DEMO_TRUSTED_LABELS, TrustedLabels


class TrustedLabelsTests(unittest.TestCase):
    def test_demo_policy_resolves_stable_ids_to_curated_names(self):
        self.assertEqual(DEMO_TRUSTED_LABELS.brand_name("brand_ate"), "ATE")
        self.assertEqual(
            DEMO_TRUSTED_LABELS.seller_name("seller_demo_parts_store"),
            "Demo Parts Store",
        )
        self.assertEqual(DEMO_TRUSTED_LABELS.brand_id_for_name("ate"), "brand_ate")
        self.assertIsNone(DEMO_TRUSTED_LABELS.brand_name("brand_unknown"))
        self.assertIsNone(DEMO_TRUSTED_LABELS.seller_name("seller_unknown"))
        self.assertIsNone(DEMO_TRUSTED_LABELS.brand_id_for_name("ATE extra"))

    def test_policy_rejects_empty_duplicate_and_malformed_configuration(self):
        with self.assertRaises(ValueError):
            TrustedLabels(brands=(), sellers=(("seller_test", "Seller"),))
        with self.assertRaises(ValueError):
            TrustedLabels(
                brands=(("brand_ate", "ATE"), ("brand_ate", "Other")),
                sellers=(("seller_test", "Seller"),),
            )
        with self.assertRaises(ValueError):
            TrustedLabels(
                brands=(("brand_ate", "ATE"), ("brand_other", "ate")),
                sellers=(("seller_test", "Seller"),),
            )
        with self.assertRaises(ValueError):
            TrustedLabels(
                brands=(("brand;inject", "ATE"),),
                sellers=(("seller_test", "Seller"),),
            )


if __name__ == "__main__":
    unittest.main()
