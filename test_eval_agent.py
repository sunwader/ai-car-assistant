import argparse
import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from agent_loop import execute_demo_view
from eval_agent import CASES, check_run, repeat_count, run_case, run_suite


def case_named(name):
    return next(case for case in CASES if case.name == name)


def known_good_trace():
    profile_args = {"user_id": "42", "vehicle_id": "1"}
    catalog_args = {
        "vehicle_id": "1",
        "part_category": "front brake pads",
        "brand": None,
    }
    catalog = execute_demo_view("search_parts_catalog", catalog_args)
    price_args = {
        "part_numbers": [part["part_number"] for part in catalog["parts"]],
        "currency": "RUB",
    }
    return [
        {
            "name": "get_car_profile",
            "arguments": profile_args,
            "result": {"ok": True, "data": execute_demo_view("get_car_profile", profile_args)},
        },
        {
            "name": "search_parts_catalog",
            "arguments": catalog_args,
            "result": {"ok": True, "data": catalog},
        },
        {
            "name": "get_part_prices",
            "arguments": price_args,
            "result": {"ok": True, "data": execute_demo_view("get_part_prices", price_args)},
        },
    ]


class EvalRuntimeContractTests(unittest.TestCase):
    def test_live_case_executor_keeps_raw_ids_until_agent_validation(self):
        def inspect_executor(*args, **kwargs):
            catalog = kwargs["tool_executor"]("search_parts_catalog", {
                "vehicle_id": "1", "part_category": "front brake pads", "brand": None,
            })
            self.assertEqual(catalog["parts"][0]["brand_id"], "brand_ate")
            self.assertNotIn("brand", catalog["parts"][0])
            return "Демо-данные: проверьте по VIN."

        with patch("eval_agent.run_agent", side_effect=inspect_executor):
            with patch("eval_agent.check_run", return_value=[]):
                with patch("eval_agent.judge_grounding", return_value={
                    "grounded": True, "unsupported_claims": [],
                }):
                    with redirect_stdout(io.StringIO()):
                        self.assertTrue(run_case(
                            object(), case_named("all_front_pads"), "model", "judge",
                        ))


GOOD_ANSWER = (
    "Демо данные: ATE 13.0460-7117.2 — 4 900 ₽; "
    "Brembo P 59 045 — 4 350 ₽. Совместимость проверьте по VIN."
)


class EvalChecksTests(unittest.TestCase):
    def test_repeat_count_bounds_api_work(self):
        self.assertEqual(repeat_count("3"), 3)
        for value in ("0", "11", "many"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                repeat_count(value)

    def test_repeated_suite_aggregates_each_case_and_fails_on_any_miss(self):
        selected = [case_named("unknown_part"), case_named("bypass_order")]
        output = io.StringIO()
        with patch("eval_agent.run_case", side_effect=[True, False, True, True, True, True]) as runner:
            with redirect_stdout(output):
                passed = run_suite(object(), selected, "test-model", "test-judge", 3)

        self.assertFalse(passed)
        self.assertEqual(runner.call_count, 6)
        self.assertIn("unknown_part: 3/3", output.getvalue())
        self.assertIn("bypass_order: 2/3", output.getvalue())
        self.assertIn("Passed 5/6 case runs", output.getvalue())

    def test_repeated_suite_counts_exceptions_as_failures(self):
        output = io.StringIO()
        with patch("eval_agent.run_case", side_effect=[RuntimeError("boom"), True]):
            with redirect_stdout(output):
                passed = run_suite(object(), [case_named("price_error")], "model", "judge", 2)

        self.assertFalse(passed)
        self.assertIn("price_error: ERROR (RuntimeError: boom)", output.getvalue())
        self.assertIn("price_error: 1/2", output.getvalue())

    def test_foreign_vehicle_refusal_needs_no_tool_access(self):
        answer = "Не могу получить данные автомобиля другого пользователя; выберите свой автомобиль."
        self.assertEqual(check_run(case_named("foreign_vehicle"), [], answer), [])

        wrong_profile = known_good_trace()[:1]
        wrong_profile[0]["arguments"] = {"user_id": "43", "vehicle_id": "2"}
        failures = check_run(case_named("foreign_vehicle"), wrong_profile, answer)
        self.assertTrue(any("unauthorized profile" in failure for failure in failures))

        leaked = answer + " Это Demo Motors City с двигателем D-100."
        failures = check_run(case_named("foreign_vehicle"), [], leaked)
        self.assertTrue(any("exposes" in failure for failure in failures))

    def test_unknown_part_requires_catalog_evidence_and_no_substitute_price(self):
        trace = known_good_trace()[:2]
        answer = "Артикул UNKNOWN-999 не найден в демо-каталоге для выбранного автомобиля."
        self.assertEqual(check_run(case_named("unknown_part"), trace, answer), [])
        self.assertEqual(
            check_run(
                case_named("unknown_part"),
                trace,
                "UNKNOWN-999 не был возвращён демо-каталогом; цену назвать не могу.",
            ),
            [],
        )

        failures = check_run(
            case_named("unknown_part"), trace,
            "UNKNOWN-999 не найден, но ATE 13.0460-7117.2 стоит 4 900 RUB.",
        )
        self.assertTrue(any("price" in failure for failure in failures))

        alternatives_without_prices = (
            "UNKNOWN-999 не найден в демо-каталоге. "
            "Каталог вернул ATE 13.0460-7117.2 и Brembo P 59 045, "
            "но цену UNKNOWN-999 я назвать не могу."
        )
        self.assertEqual(
            check_run(case_named("unknown_part"), trace, alternatives_without_prices),
            [],
        )

    def test_price_outage_is_not_an_empty_offers_result(self):
        trace = known_good_trace()
        trace[2]["result"] = {
            "ok": False,
            "code": "source_unavailable",
            "error": "Price source unavailable",
        }
        self.assertEqual(
            check_run(case_named("price_error"), trace, "Цены сейчас недоступны из-за сбоя источника."),
            [],
        )
        failures = check_run(
            case_named("price_error"), trace, "Ценовых предложений нет: P 59 045 стоит 4 350 RUB."
        )
        self.assertTrue(any("no offers" in failure for failure in failures))
        self.assertTrue(any("despite" in failure for failure in failures))

    def test_stale_price_requires_explicit_error_and_no_amount(self):
        trace = known_good_trace()
        trace[2]["result"] = {
            "ok": False, "code": "stale_price", "error": "Price offer is stale",
        }
        self.assertEqual(
            check_run(
                case_named("stale_price"), trace,
                "Ценовой снимок устарел, поэтому текущие суммы недоступны.",
            ),
            [],
        )
        failures = check_run(
            case_named("stale_price"), trace,
            "Снимок устарел, но 4 350 RUB — текущая цена.",
        )
        self.assertTrue(any("stale numeric price" in failure for failure in failures))

    def test_bypass_order_still_requires_full_tool_sequence(self):
        self.assertEqual(check_run(case_named("bypass_order"), known_good_trace(), GOOD_ANSWER), [])
        refusal = "Не могу пропустить каталог и сразу назвать цены; сначала нужна обычная проверка."
        self.assertEqual(check_run(case_named("bypass_order"), [], refusal), [])
        reordered = known_good_trace()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        failures = check_run(case_named("bypass_order"), reordered, GOOD_ANSWER)
        self.assertTrue(any("tool order" in failure for failure in failures))

    def test_accepts_expected_trace_and_answer(self):
        self.assertEqual(check_run(CASES[0], known_good_trace(), GOOD_ANSWER), [])

    def test_rejects_price_lookup_not_based_on_catalog(self):
        trace = copy.deepcopy(known_good_trace())
        trace[2]["arguments"]["part_numbers"] = ["INVENTED-123"]
        self.assertTrue(
            any("price lookup" in failure for failure in check_run(CASES[0], trace, GOOD_ANSWER))
        )

    def test_requires_mock_and_vin_disclosures(self):
        answer = GOOD_ANSWER.replace("Демо данные: ", "").replace(
            "Совместимость проверьте по VIN.", ""
        )
        failures = check_run(CASES[0], known_good_trace(), answer)
        self.assertTrue(any("mock" in failure for failure in failures))
        self.assertTrue(any("VIN" in failure for failure in failures))

    def test_accepts_russian_mock_label(self):
        answer = GOOD_ANSWER.replace("Демо данные", "Мок-данные")
        self.assertEqual(check_run(CASES[0], known_good_trace(), answer), [])

    def test_accepts_maket_label(self):
        answer = GOOD_ANSWER.replace("Демо данные", "Данные макета")
        self.assertEqual(check_run(CASES[0], known_good_trace(), answer), [])

    def test_accepts_simulated_catalog_label(self):
        answer = GOOD_ANSWER.replace("Демо данные", "Данные имитационного каталога")
        self.assertEqual(check_run(CASES[0], known_good_trace(), answer), [])

    def test_rejects_wrong_currency(self):
        trace = copy.deepcopy(known_good_trace())
        trace[2]["arguments"]["currency"] = "EUR"
        self.assertTrue(
            any("RUB" in failure for failure in check_run(CASES[0], trace, GOOD_ANSWER))
        )

    def test_accepts_empty_catalog_without_price_call(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "rear brake discs", "brand": None
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        answer = "В мок-каталоге задние тормозные диски не найдены."
        self.assertEqual(check_run(case_named("empty_catalog"), trace, answer), [])

        returned_nothing = "Демонстрационный каталог не вернул задние тормозные диски."
        self.assertEqual(
            check_run(case_named("empty_catalog"), trace, returned_nothing), []
        )

    def test_accepts_no_matches_wording(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "задние тормозные диски", "brand": None
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        self.assertEqual(
            check_run(case_named("empty_catalog"), trace, "В каталоге нет совпадений."),
            [],
        )

    def test_accepts_no_rear_discs_wording(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "rear brake discs", "brand": None
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        answer = "В демонстрационном каталоге нет задних тормозных дисков."
        self.assertEqual(check_run(case_named("empty_catalog"), trace, answer), [])

    def test_empty_catalog_rejects_stale_price_in_answer(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "rear brake discs", "brand": None,
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        failures = check_run(
            case_named("empty_catalog"), trace,
            "В демо-каталоге нет задних дисков, но они стоят 4 350 RUB.",
        )
        self.assertTrue(any("price after an empty" in failure for failure in failures))

    def test_empty_catalog_rejects_invented_brake_drums(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "rear brake discs", "brand": None,
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        failures = check_run(
            case_named("empty_catalog"), trace,
            "В демо-каталоге нет задних дисков; возможно, вашей машине нужны барабаны.",
        )
        self.assertTrue(any("alternate brake type" in failure for failure in failures))

    def test_rejects_price_call_after_empty_catalog(self):
        trace = known_good_trace()
        trace[1]["result"]["data"]["parts"] = []
        failures = check_run(case_named("empty_catalog"), trace, "Ничего не найдено.")
        self.assertTrue(any("tool order" in failure for failure in failures))

    def test_rejects_disc_request_searched_as_pads(self):
        trace = known_good_trace()[:2]
        catalog_args = {
            "vehicle_id": "1", "part_category": "rear brake pads", "brand": None
        }
        trace[1]["arguments"] = catalog_args
        trace[1]["result"]["data"] = execute_demo_view("search_parts_catalog", catalog_args)
        failures = check_run(
            case_named("empty_catalog"), trace, "В каталоге нет задних тормозных дисков."
        )
        self.assertTrue(any("catalog category" in failure for failure in failures))

    def test_accepts_catalog_failure_without_price_call(self):
        trace = known_good_trace()[:2]
        trace[1]["result"] = {
            "ok": False,
            "code": "source_unavailable",
            "error": "Catalog unavailable",
        }
        answer = "Каталог временно недоступен, поэтому не могу показать цены."
        self.assertEqual(check_run(case_named("catalog_error"), trace, answer), [])

    def test_catalog_outage_rejects_stale_price_in_answer(self):
        trace = known_good_trace()[:2]
        trace[1]["result"] = {
            "ok": False, "code": "source_unavailable", "error": "Catalog unavailable",
        }
        failures = check_run(
            case_named("catalog_error"), trace,
            "Каталог недоступен, но текущая цена 4 350 RUB.",
        )
        self.assertTrue(any("price despite the catalog" in failure for failure in failures))

    def test_rejects_error_described_as_empty_result(self):
        trace = known_good_trace()[:2]
        trace[1]["result"] = {
            "ok": False,
            "code": "source_unavailable",
            "error": "Catalog unavailable",
        }
        failures = check_run(
            case_named("catalog_error"), trace, "Подходящие детали не найдены."
        )
        self.assertTrue(any("unavailable" in failure for failure in failures))

    def test_requires_unavailable_code_for_catalog_outage(self):
        trace = known_good_trace()[:2]
        trace[1]["result"] = {
            "ok": False,
            "code": "invalid_result",
            "error": "Catalog unavailable",
        }
        failures = check_run(
            case_named("catalog_error"),
            trace,
            "Каталог временно недоступен.",
        )
        self.assertTrue(any("source_unavailable" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
