# AI Car Assistant | AI Automation case study

## The problem

An LLM can sound confident about vehicle parts and prices even when the
catalog is empty, a price source is unavailable, or an offer is old. A useful
assistant needs to connect the model to data while keeping source access and
answer claims under application control.

## What the automation does

For a selected demo vehicle, the assistant loads its profile, searches a
read-only parts catalog, then requests prices only for part numbers returned
by that catalog. The model may decide which tool to request next, but the
application authorizes each call and validates its result. The final answer
must not invent a current price after a failed or stale lookup.

```text
User request → Anthropic Messages API → tool_use
                                      ↓
                    per-turn authorization and tool registry
                                      ↓
                          mock or read-only SQLite source
                                      ↓
                    validated tool_result → model → answer
```

## Engineering decisions

- **Direct tool loop:** the orchestration, iteration cap, tool-call cap, and
  session history are visible in ordinary Python rather than hidden in an
  agent framework. This makes failures and control flow inspectable.
- **Application-owned authority:** the selected vehicle and approved brand /
  seller labels come from the application, not user text or source-provided
  display strings. Model-requested IDs and part numbers are checked before
  any source call.
- **Data-source boundary:** mock and SQLite implement the same three read-only
  operations. Shared contract tests exercise the application's validator for
  both. No external parts provider is claimed.
- **Explicit failure states:** an empty catalog, no price offers, a source
  outage, malformed data, and a stale offer have different outcomes. Offers
  need a source observation time no more than 15 minutes old in this demo.
- **Grounding and privacy controls:** validated fields are projected before
  the model sees them; free-text notes and extra metadata are omitted. Logs
  exclude tool arguments and source results. These are bounded safeguards,
  not a guarantee against every prompt injection or false claim.

## See it work

From this directory, run `python demo_offline.py`. It needs no API key or
external service. The four scenarios show the tool sequence and outcome.
The model calls and final wording are scripted, while the real dispatcher,
source adapters, authorization checks, and result validation execute.

For a live LLM run, install `requirements.txt`, provide
`ANTHROPIC_API_KEY` through the environment or local `.env`, and run
`python agent_loop.py "Подбери передние тормозные колодки"`. A live run
uses paid Anthropic API calls. The full setup and evaluation commands are
in [`README.md`](README.md).

## Evidence and scope

The offline regression suite contains unit, integration, contract, and
evaluation-harness tests. The evaluation plan also records targeted live
smoke tests and observed failures; these are not statistical reliability
estimates. Run `python -m unittest -q` to see the current test result and
read [`EVAL_PLAN.md`](EVAL_PLAN.md) for the scenario matrix.

This is a portfolio-ready **automation prototype**, not a production
automotive or commerce system. The two vehicles, catalog entries, and prices
are fictional. CLI vehicle selection is not user authentication, VIN fitment
is unverified, and no real seller or provider is connected. A production
version would need authenticated vehicle ownership, a documented provider
contract, provider-specific freshness rules, persistent isolated sessions,
and operational monitoring before purchase guidance or ordering.
