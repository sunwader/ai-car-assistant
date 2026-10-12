# Agent Tool Loop evaluation baseline

Date: 2026-09-23  
Generator: `claude-haiku-4-5-20251001`  
Grounding judge: `claude-sonnet-5`

## Success criteria

| Area | Check | Baseline target |
| --- | --- | --- |
| Tool orchestration | Profile → catalog → prices, with catalog part numbers passed to the price tool | 2/2 cases |
| Tool contract | Mock prices requested and returned in RUB | 2/2 cases |
| Answer completeness | Offered part numbers and prices, mock label, VIN reminder | 2/2 cases |
| Grounding | Judge finds no factual claims unsupported by tool results | 2/2 cases |
| Judge calibration | Judge rejects a known false brand and part-lifespan claim | Pass |

## Result

The judge calibration passed. The judge identified both unsupported claims:
“проверенный немецкий бренд” and “ресурсом 80 000 км”.

The final live run passed both cases:

```text
all_front_pads: PASS
  get_car_profile -> search_parts_catalog -> get_part_prices
ate_front_pads: PASS
  get_car_profile -> search_parts_catalog -> get_part_prices
Passed 2/2 cases
```

The offline suite passed 12 tests.

## What this evaluation exposed

The first evaluator rejected a valid Russian mock label (“мок-данные”) because
its string rule was too narrow. It also exposed a real contract error: the model
could ask the mock price tool for EUR even though its numeric prices were fixed
RUB examples. The tool schema and implementation now permit RUB only, and the
evaluator checks the currency.

## Limits and next coverage

Two prompts do not establish reliability across other parts, vehicles, missing
catalog results, or ambiguous user requests. The LLM judge can also make
mistakes; its calibration checks one known failure mode, not every possible
unsupported claim. Before using this as a release gate, expand to at least 20
representative cases, include negative and empty-result paths, and manually
review a sample of judge decisions.
