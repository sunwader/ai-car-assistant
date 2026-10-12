"""Live multi-turn evaluation of context, outages, and fresh tool use."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from typing import Any

from agent_loop import (
    AgentSession,
    DEFAULT_MODEL,
    configure_console,
    create_anthropic_client,
    execute_demo_view,
    make_tool_executor,
    run_agent,
    ToolExecutionError,
)
from eval_agent import (
    DEFAULT_JUDGE_MODEL,
    EvalCase,
    calibrate_judge,
    check_run,
    judge_grounding,
    repeat_count,
)
from mock_data import MockCarDataSource


BRAND_REFINEMENT = (
    EvalCase(
        name="initial_front_pads",
        prompt="Подбери передние тормозные колодки для моей машины и покажи все цены.",
        scenario="offers",
        expected_brand=None,
        expected_parts=2,
        expected_category="front brake pads",
    ),
    EvalCase(
        name="followup_brembo",
        prompt="Оставь только Brembo и покажи цену на эти колодки.",
        scenario="offers",
        expected_brand="Brembo",
        expected_parts=1,
        expected_category="front brake pads",
    ),
    EvalCase(
        name="followup_current_price",
        prompt="Сколько стоят выбранные колодки сейчас? Перепроверь цену.",
        scenario="offers",
        expected_brand="Brembo",
        expected_parts=1,
        expected_category="front brake pads",
    ),
)


SWITCH_CATEGORY = (
    BRAND_REFINEMENT[0],
    EvalCase(
        name="switch_to_rear_discs",
        prompt="Теперь подбери задние тормозные диски и покажи цены, если они есть.",
        scenario="empty",
        expected_brand=None,
        expected_parts=0,
        expected_category="rear brake discs",
    ),
    EvalCase(
        name="rear_discs_price_followup",
        prompt="А сколько стоят выбранные задние диски? Проверь ещё раз.",
        scenario="empty",
        expected_brand=None,
        expected_parts=0,
        expected_category="rear brake discs",
    ),
)

CATALOG_OUTAGE = (
    BRAND_REFINEMENT[0],
    EvalCase(
        name="catalog_outage_followup",
        prompt="Проверь каталог передних колодок ещё раз и скажи, есть ли варианты.",
        scenario="catalog_error",
        expected_brand=None,
        expected_parts=None,
        expected_category="front brake pads",
    ),
)

PRICE_OUTAGE = (
    BRAND_REFINEMENT[0],
    EvalCase(
        name="price_outage_followup",
        prompt="Проверь текущие цены на эти передние колодки ещё раз.",
        scenario="price_error",
        expected_brand=None,
        expected_parts=2,
        expected_category="front brake pads",
    ),
)

DIALOGUES = {
    "brand_refinement": BRAND_REFINEMENT,
    "switch_category": SWITCH_CATEGORY,
    "catalog_outage": CATALOG_OUTAGE,
    "price_outage": PRICE_OUTAGE,
}


def run_dialogue(
    client: Any,
    model: str,
    judge_model: str,
    *,
    name: str = "brand_refinement",
) -> bool:
    cases = DIALOGUES[name]
    session = AgentSession(user_id="42", vehicle_id="1", max_turns=len(cases))
    source = MockCarDataSource()
    base_executor = make_tool_executor(source)
    previous_turns: list[dict[str, Any]] = []
    passed = 0
    print(f"Dialogue {name}: {len(cases)} turns")
    for case in cases:
        trace: list[dict[str, Any]] = []

        def eval_executor(
            tool_name: str, arguments: dict[str, Any], scenario: str = case.scenario
        ) -> dict[str, Any]:
            if scenario == "catalog_error" and tool_name == "search_parts_catalog":
                raise ToolExecutionError(
                    "Catalog temporarily unavailable (eval fault)",
                    code="source_unavailable",
                )
            if scenario == "price_error" and tool_name == "get_part_prices":
                raise ToolExecutionError(
                    "Price source temporarily unavailable (eval fault)",
                    code="source_unavailable",
                )
            return base_executor(tool_name, arguments)

        try:
            answer = run_agent(
                case.prompt, client=client, model=model, session=session,
                tool_executor=eval_executor, trace=trace,
            )
            failures = check_run(case, trace, answer)
            verdict = judge_grounding(
                client, model=judge_model, question=case.prompt,
                trace=trace, answer=answer, previous_turns=deepcopy(previous_turns),
            )
            previous_turns.append({
                "question": case.prompt,
                "tool_outputs": [
                    {"tool": event["name"], "result": event["result"]}
                    for event in trace
                ],
            })
            unsupported = verdict["unsupported_claims"]
            success = not failures and verdict["grounded"] and not unsupported
            print(f"{case.name}: {'PASS' if success else 'FAIL'}")
            print("  tools: " + " -> ".join(event["name"] for event in trace))
            for failure in failures:
                print(f"  deterministic: {failure}")
            for claim in unsupported:
                print(f"  unsupported claim: {claim}")
            if not verdict["grounded"] and not unsupported:
                print("  judge marked the answer ungrounded without a specific claim")
            if not success:
                print(f"  answer: {answer}")
            passed += bool(success)
        except Exception as exc:
            print(f"{case.name}: ERROR ({type(exc).__name__}: {exc})")
            break
    print(f"Passed {passed}/{len(cases)} turns")
    return passed == len(cases)


def calibrate_dialogue_judge(client: Any, model: str) -> bool:
    """Distinguish valid historical claims from stale current-price claims."""
    profile = execute_demo_view("get_car_profile", {"user_id": "42", "vehicle_id": "1"})
    catalog = execute_demo_view("search_parts_catalog", {
        "vehicle_id": "1", "part_category": "front brake pads", "brand": None,
    })
    pricing = execute_demo_view("get_part_prices", {
        "part_numbers": ["13.0460-7117.2", "P 59 045"], "currency": "RUB",
    })
    previous_turns = [{
        "question": "Подбери передние колодки и цены.",
        "tool_outputs": [
            {"tool": "get_car_profile", "result": {"ok": True, "data": profile}},
            {"tool": "search_parts_catalog", "result": {"ok": True, "data": catalog}},
            {"tool": "get_part_prices", "result": {"ok": True, "data": pricing}},
        ],
    }]
    current_trace = [
        {"name": "get_car_profile", "result": {"ok": True, "data": profile}},
        {"name": "search_parts_catalog", "result": {
            "ok": False, "code": "source_unavailable", "error": "Catalog unavailable",
        }},
    ]
    valid = judge_grounding(
        client, model=model,
        question="Проверь каталог передних колодок ещё раз.",
        trace=current_trace, previous_turns=previous_turns,
        answer=(
            "В предыдущем поиске демо-каталог вернул ATE и Brembo, "
            "но сейчас каталог недоступен; наличие не подтверждено."
        ),
    )
    invalid = judge_grounding(
        client, model=model,
        question="Проверь каталог передних колодок ещё раз.",
        trace=current_trace, previous_turns=previous_turns,
        answer="Сейчас Brembo P 59 045 стоит 4 350 RUB и есть в наличии.",
    )
    passed = (
        valid["grounded"] and not valid["unsupported_claims"]
        and not invalid["grounded"] and bool(invalid["unsupported_claims"])
    )
    print(f"dialogue_judge_calibration: {'PASS' if passed else 'FAIL'}")
    for claim in valid["unsupported_claims"]:
        print(f"  unexpected historical rejection: {claim}")
    for claim in invalid["unsupported_claims"]:
        print(f"  stale current claim: {claim}")
    return bool(passed)


def run_dialogue_suite(
    client: Any, model: str, judge_model: str,
    selected: list[str], repeats: int,
) -> bool:
    """Each repetition starts a new session; one failed turn fails its dialogue."""
    total = len(selected) * repeats
    total_turns = repeats * sum(len(DIALOGUES[name]) for name in selected)
    print(f"Planned {total_turns} live turns across {total} dialogue run(s)")
    passed_by_dialogue = {name: 0 for name in selected}
    for repetition in range(1, repeats + 1):
        if repeats > 1:
            print(f"Repetition {repetition}/{repeats}")
        for name in selected:
            try:
                passed_by_dialogue[name] += bool(
                    run_dialogue(client, model, judge_model, name=name)
                )
            except Exception as exc:
                print(f"{name}: ERROR ({type(exc).__name__}: {exc})")
    if repeats > 1:
        print("Per-dialogue results:")
        for name in selected:
            print(f"  {name}: {passed_by_dialogue[name]}/{repeats}")
    passed = sum(passed_by_dialogue.values())
    print(f"Passed {passed}/{total} dialogue runs")
    return passed == total


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--judge-model", default=os.getenv("ANTHROPIC_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    )
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument(
        "--repeat", type=repeat_count, default=1,
        help="Repeat each selected whole dialogue 1-10 times; default: 1",
    )
    parser.add_argument(
        "--dialogue", action="append", choices=sorted(DIALOGUES),
        help="Select a dialogue; repeat to run several (default: brand_refinement)",
    )
    args = parser.parse_args()
    if args.calibrate_only and args.skip_calibration:
        parser.error("--calibrate-only and --skip-calibration cannot be combined")
    selected = args.dialogue or ["brand_refinement"]
    client = create_anthropic_client()
    if args.calibrate_only:
        return 0 if calibrate_dialogue_judge(client, args.judge_model) else 1
    if not args.skip_calibration:
        if not calibrate_judge(client, args.judge_model):
            return 1
        if not calibrate_dialogue_judge(client, args.judge_model):
            return 1
    return 0 if run_dialogue_suite(
        client, args.model, args.judge_model, selected, args.repeat,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
