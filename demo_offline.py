"""Replay the real tool loop with scripted model decisions, without an API key."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

from agent_loop import configure_console, run_agent
from car_tools import DataSourceUnavailableError
from mock_data import MockCarDataSource


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def _response(*tool_calls: SimpleNamespace, text: str = "") -> SimpleNamespace:
    content = list(tool_calls)
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        id="offline_replay",
        content=content,
        stop_reason="tool_use" if tool_calls else "end_turn",
    )


class _ReplayMessages:
    def __init__(self, responses: Sequence[SimpleNamespace]) -> None:
        self._responses = iter(responses)

    def create(self, **request: Any) -> SimpleNamespace:
        try:
            return next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("Offline replay ran out of scripted model responses") from exc


class _ReplayClient:
    def __init__(self, responses: Sequence[SimpleNamespace]) -> None:
        self.messages = _ReplayMessages(responses)


class _UnavailablePriceSource(MockCarDataSource):
    def get_part_prices(self, part_numbers: list[str], currency: str) -> dict[str, Any]:
        raise DataSourceUnavailableError("Simulated provider outage")


class _StalePriceSource(MockCarDataSource):
    def get_part_prices(self, part_numbers: list[str], currency: str) -> dict[str, Any]:
        result = super().get_part_prices(part_numbers, currency)
        for offer in result["offers"]:
            offer["observed_at"] = "2000-01-01T00:00:00Z"
        return result


@dataclass(frozen=True)
class DemoScenario:
    prompt: str
    category: str
    part_numbers: tuple[str, ...]
    final_answer: str
    source: MockCarDataSource


SCENARIOS = {
    "success": DemoScenario(
        prompt="Подбери передние тормозные колодки и покажи цены.",
        category="front brake pads",
        part_numbers=("13.0460-7117.2", "P 59 045"),
        final_answer=(
            "Демо-каталог: ATE 13.0460-7117.2 — 4 900 RUB, "
            "Brembo P 59 045 — 4 350 RUB. Совместимость проверьте по VIN."
        ),
        source=MockCarDataSource(),
    ),
    "empty_catalog": DemoScenario(
        prompt="Найди задние тормозные диски для моей машины.",
        category="rear brake discs",
        part_numbers=(),
        final_answer="Демо-каталог не вернул задних дисков; цену назвать нельзя.",
        source=MockCarDataSource(),
    ),
    "price_outage": DemoScenario(
        prompt="Подбери передние колодки и проверь цены.",
        category="front brake pads",
        part_numbers=("13.0460-7117.2", "P 59 045"),
        final_answer="Демо-каталог нашёл колодки, но источник цен недоступен; суммы не подтверждены.",
        source=_UnavailablePriceSource(),
    ),
    "stale_price": DemoScenario(
        prompt="Проверь цены передних колодок сейчас.",
        category="front brake pads",
        part_numbers=("13.0460-7117.2", "P 59 045"),
        final_answer="Снимок демо-цен устарел; актуальную сумму назвать нельзя.",
        source=_StalePriceSource(),
    ),
}


def run_scenario(name: str) -> tuple[str, list[dict[str, Any]]]:
    """Exercise the production dispatcher; only the model decisions are replayed."""
    case = SCENARIOS[name]
    responses = [
        _response(_tool_call(
            "get_car_profile", {"user_id": "42", "vehicle_id": "1"}, "profile",
        )),
        _response(_tool_call(
            "search_parts_catalog", {
                "vehicle_id": "1", "part_category": case.category, "brand": None,
            }, "catalog",
        )),
    ]
    if case.part_numbers:
        responses.append(_response(_tool_call(
            "get_part_prices", {
                "part_numbers": list(case.part_numbers), "currency": "RUB",
            }, "prices",
        )))
    responses.append(_response(text=case.final_answer))

    trace: list[dict[str, Any]] = []
    answer = run_agent(
        case.prompt, client=_ReplayClient(responses),
        data_source=case.source, trace=trace,
    )
    return answer, trace


def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    args = parser.parse_args(argv)
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    print("Offline replay: model decisions and final wording are scripted; no API key is used.")
    for name in names:
        answer, trace = run_scenario(name)
        print(f"\n{name}: {SCENARIOS[name].prompt}")
        for event in trace:
            result = event["result"]
            outcome = "ok" if result["ok"] else result["code"]
            print(f"  {event['name']} -> {outcome}")
        print(f"  Ответ: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
