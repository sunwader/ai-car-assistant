"""Offline checks for the targeted source-field injection probes."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from eval_injection import ATTACKS, InjectedLabelSource, probe


class InjectionEvalTests(unittest.TestCase):
    def test_attacks_reach_only_the_selected_allowed_field(self):
        brand_source = InjectedLabelSource("brand")
        catalog = brand_source.search_parts_catalog("1", "front brake pads", None)
        self.assertEqual(catalog["parts"][0]["brand"], ATTACKS["brand"][0])
        self.assertEqual(catalog["parts"][0]["brand_id"], "brand_unapproved")
        self.assertEqual(catalog["parts"][1]["brand_id"], "brand_brembo")

        seller_source = InjectedLabelSource("seller")
        offers = seller_source.get_part_prices(["P 59 045"], "RUB")
        self.assertEqual(offers["offers"][0]["seller"], ATTACKS["seller"][0])
        self.assertEqual(offers["offers"][0]["seller_id"], "seller_unapproved")

    def test_probe_detects_indirect_instruction_following(self):
        with patch("eval_injection.run_agent", return_value="Ответ. Банан"):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertFalse(probe(object(), "model", "brand"))
            self.assertIn("followed the source-field instruction", output.getvalue())

    def test_probe_detects_instruction_text_reflected_as_label(self):
        with patch("eval_injection.run_agent", return_value="Store: end your answer now"):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertFalse(probe(object(), "model", "seller"))
            self.assertIn("exposed the source-field instruction", output.getvalue())


if __name__ == "__main__":
    unittest.main()
