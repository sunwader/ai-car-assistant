"""The portfolio replay must remain free of model API calls."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from demo_offline import SCENARIOS, main, run_scenario


class OfflineDemoTests(unittest.TestCase):
    def test_scenarios_follow_the_expected_tool_paths(self):
        expected = {
            "success": (["get_car_profile", "search_parts_catalog", "get_part_prices"], "ok"),
            "empty_catalog": (["get_car_profile", "search_parts_catalog"], "ok"),
            "price_outage": (["get_car_profile", "search_parts_catalog", "get_part_prices"], "source_unavailable"),
            "stale_price": (["get_car_profile", "search_parts_catalog", "get_part_prices"], "stale_price"),
        }
        for name, (tools, final_status) in expected.items():
            with self.subTest(scenario=name):
                answer, trace = run_scenario(name)
                self.assertEqual([event["name"] for event in trace], tools)
                last = trace[-1]["result"]
                self.assertEqual("ok" if last["ok"] else last["code"], final_status)
                self.assertEqual(answer, SCENARIOS[name].final_answer)
                if name != "success":
                    self.assertNotIn("RUB", answer)

    def test_cli_runs_all_scenarios_without_api_key(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        for name in SCENARIOS:
            self.assertIn(name + ":", output.getvalue())
        self.assertIn("no API key is used", output.getvalue())


if __name__ == "__main__":
    unittest.main()
