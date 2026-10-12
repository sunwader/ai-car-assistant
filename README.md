# AI Car Assistant: a tool-based AI automation case study

A read-only assistant that selects a demo vehicle, finds parts, checks offers,
and declines to quote missing or stale prices. The project demonstrates an
AI automation workflow with explicit tool authorization, replaceable data
sources, error handling, and evaluation. It is a portfolio prototype, not a
verified parts service or an ordering system. See the short
[`PORTFOLIO.md`](PORTFOLIO.md) case study for the problem, design, evidence, and
limitations.

## Try it in under a minute, no API key

```text
python demo_offline.py
```

This replays four scripted model decisions through the **actual application
tool loop**: a successful lookup, an empty catalog, a price-source outage,
and a stale price. It does not call Anthropic or evaluate LLM reasoning. To
run one case, use `python demo_offline.py --scenario stale_price`. Python 3.11+
is sufficient for this offline replay; the live version below needs the
Anthropic SDK and an API key.

```text
success: ...
  get_car_profile -> ok
  search_parts_catalog -> ok
  get_part_prices -> ok
stale_price: ...
  get_car_profile -> ok
  search_parts_catalog -> ok
  get_part_prices -> stale_price
```

For an offline regression check after installing dependencies:

```text
python -m unittest -q
```

The live implementation uses the Anthropic Messages API directly, with no
agent framework. The detailed architecture and setup follow.

## Architecture

```text
User request
    -> Messages API
    -> tool_use
    -> per-run authorization / registry / executor
    -> car_tools.py -> selected data source (mock or SQLite)
    -> tool_result
    -> Messages API (repeat or final text)
```

`TOOL_REGISTRY` contains the model-visible `input_schema` definitions, independent
of the data source. `run_agent()` assembles one turn, logs each tool call, and
stops after a configurable number of model calls. `AgentSession` retains
completed turns for follow-ups. Tool results use the matching `tool_use_id`.

`CarTools` in `car_tools.py` exposes the three tool functions and delegates to
the `CarDataSource` contract. `MockCarDataSource` in `mock_data.py` owns the two
demo vehicles, catalog entries, prices, category matching, and mock responses.
It rejects a profile request when the user and vehicle IDs do not form a known
pair, and it scopes catalog matches by vehicle.
Unknown part numbers return no offer; the mock no longer invents a fallback
price. A mixed price request may return fewer offers than requested.
`SQLiteCarDataSource` in `sqlite_data.py` reads a local demo database through
parameterized, read-only queries. Its schema and fictional seed records are
in `demo_sqlite.sql`; no external database server or new package is needed.
`make_tool_executor(source)` binds one source to one executor. Pass
`data_source=source` to `run_agent()` to select it for that run; otherwise it
creates a fresh mock source. The CLI configures its mock source explicitly.
Two runs can therefore use different sources without changing global state or
the model-facing schemas. `execute_tool()` remains a convenience function for
standalone raw-source calls with fresh mock data; `execute_demo_view()` builds
the projected form used by evaluation examples.

`TrustedLabels` in `trusted_labels.py` is application-owned configuration,
not data returned by a tool. The mock and new SQLite fixtures return stable
`brand_id` and `seller_id` values. After validating those IDs, `RunState`
resolves them to canonical display names from the policy; raw source-provided
`brand` and `seller` text is ignored. Unknown IDs return `invalid_result`.
Existing name-based `demo.sqlite3` files remain readable: the SQLite adapter
maps recognized old names to the same IDs without rewriting the file. Older
files without `offers.observed_at` can show catalog entries, but their prices
are rejected because their age is unknown.
A new source with different IDs must pass an explicit `TrustedLabels` policy
to `run_agent()`. This protects labels, not part compatibility, seller
identity, or prices. The external-provider boundary is in
[`PROVIDER_TRUST.md`](PROVIDER_TRUST.md).

`SelectedVehicleContext` validates application-selected IDs before a model
request. They go in the system instructions; the user request remains an
unmodified user message. The context is a demo selection, not evidence of
login. The boundary and a production handoff are described in
[`CONTEXT_BOUNDARY.md`](CONTEXT_BOUNDARY.md).

`RunState` holds the selected user and vehicle IDs plus results from this run.
Before each tool call, the application checks that profile and catalog IDs match
this application-selected context, the profile was loaded before a catalog search, and prices
are requested only for part numbers from the latest nonempty catalog result.
Every new catalog search clears the previous price allowlist, including when
the search fails. Rejected calls return a tool error to the model, so it can
recover without the forbidden Python function running. A tool schema describes
the shape of a call; it is not authorization.
The API request advertises only tools useful at the current stage: profile
first, catalog after a valid profile, and prices after a nonempty catalog.
After a source outage it advertises no further tools for that turn. This
reduces mistaken calls but does not replace executor-side authorization.

`AgentSession` keeps completed turns in memory for follow-up questions. It is
bound to one user and selected vehicle, and its history is committed only
after a final answer. Each turn creates a new `RunState`: a previous catalog
result never authorizes a price call in a later turn. Sessions are limited to
six completed turns by default; `/reset` discards their history without
changing the selected user or vehicle. This is a local, single-process demo,
not persistent storage or a multi-user session service.
By default, the application refuses to finalize a session turn without a
current-turn profile and catalog lookup, or with a numeric RUB price unsupported
by a current-turn price lookup. It prompts the model to recover within the
iteration limit; a genuine source outage may instead be explained without a
price. This conservative check can add calls even for simple follow-ups.
For both one-shot and session runs, the application also rejects numeric RUB
prices without a successful current-turn price result and known unsupported
claims about alternative brakes or temporary stock after an empty catalog.
Numeric prices must also be explicitly labeled as mock/demo data.
These are targeted safeguards, not a complete semantic fact checker.

## Result and error contract

The application checks source results before sending them to the model:
profiles must match the selected IDs and include vehicle fields; catalog
results must match the requested vehicle and category and contain valid part
records; price offers must reference requested catalog part numbers and have a
known `seller_id`, nonnegative finite numeric price, RUB currency, boolean stock
status, and a timezone-aware `observed_at` timestamp. The observation must be
no older than 15 minutes and no more than 2 minutes in the future. A stale
offer fails the whole price result with `stale_price`; malformed or missing
timestamps return `invalid_result`. A recent tool call does not refresh an old
offer. Source text/IDs must be printable, nonempty, and bounded; approved
display names are bounded in the application registry. Malformed data returns `invalid_result` without
echoing the source text to the model. A price source may return fewer offers
than requested, including zero.

| Tool result | Meaning |
| --- | --- |
| `{"ok": true, "data": {"parts": []}}` | The catalog was queried and found no mock matches. |
| `{"ok": true, "data": {"offers": []}}` | The price source answered but returned no offers. |
| `{"ok": false, "code": "source_unavailable", "error": "..."}` | The source could not answer; absence of matches or offers is not established. |
| `code="invalid_result"` | The source answered with data that failed the contract. |
| `code="stale_price"` | An offer's source observation is too old to quote as a current price. |
| `code="invalid_request"` | The call violates the tool or per-run rules. |
| `code="unknown_tool"` | The model requested an unregistered tool. |
| `code="tool_failed"` | An unexpected execution error occurred. |

A future source should raise `DataSourceUnavailableError` for outages and
`DataSourceInvalidRequestError` for unsupported requests. Other exceptions are
treated as unexpected failures. The model receives a safe error message, while
unexpected exceptions are logged for debugging. Expected outages are logged
without exposing their details in the model response.

The mock tools are read-only:

- `get_car_profile`
- `search_parts_catalog`
- `get_part_prices`

The mock price tool supports RUB only. Its numeric values are not exchange
rates and must not be relabeled as another currency.

## Run

Python 3.11 or newer is recommended. If `ANTHROPIC_API_KEY` is already set in
your environment, run:

```powershell
.\.venv\Scripts\python.exe agent_loop.py --debug "Подбери передние тормозные колодки"
```

Alternatively, copy `.env.example` to `.env`, put your Anthropic API key in the
local file, and load it for a single run:

```powershell
Copy-Item .env.example .env
notepad .env
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv run --env-file .env .venv\Scripts\python.exe agent_loop.py --debug "Подбери передние тормозные колодки"
```

Do not paste the key into a chat or commit `.env`. Change the model using
`ANTHROPIC_MODEL` and the safety limit using `AGENT_MAX_ITERATIONS`.
The CLI also limits each turn to 8 model calls, 16 tool calls, and 50,000
reported tokens by default; use `--max-iterations`, `--max-tool-calls`, and
`--max-turn-tokens` to change them. `--request-timeout` sets a 60-second
per-API-call timeout by default. SDK automatic retries are disabled, so a
network failure is visible rather than silently adding calls. The token
counter includes reported input, output, and cache tokens, but is checked
*after* each response: it may exceed the threshold by one call and is not an
exact monetary spending cap. The normal application logs contain tool names,
durations, outcomes, and token counts, but omit tool arguments and results.

To try the SQLite source, first create the local demo database explicitly:

```powershell
.\.venv\Scripts\python.exe sqlite_data.py demo.sqlite3
.\.venv\Scripts\python.exe agent_loop.py --sqlite-db demo.sqlite3 "Подбери передние тормозные колодки"
```

The initializer refuses to overwrite an existing database. Newly created
SQLite demo offers receive `observed_at` when the file is initialized and
therefore expire after 15 minutes; create a new file at a different path for
a fresh demo. Do not overwrite an existing file merely to refresh prices.
Legacy files without `observed_at` remain readable, but cannot support a
numeric price. `demo.sqlite3` is ignored by Git. The second demo vehicle can be
selected with `--user-id 43 --vehicle-id 2` for either local source. Its
front-pad part has no price offer; its rear-disc part has one. SQLite reads
use `mode=ro`, so the agent cannot edit the file.

For follow-up questions in one terminal session:

```powershell
.\.venv\Scripts\python.exe agent_loop.py --chat --sqlite-db demo.sqlite3
```

Ask a question, then refine it (for example, "только Brembo"). Enter `/reset`
to start a fresh conversation for the same selected vehicle, or `/exit` to
quit. `--max-turns` changes the history cap. The agent re-checks the profile
and catalog for each new parts or price request; it does not treat old offers
as current evidence. An unfinished or failed turn does not alter the saved
history. The CLI's user and vehicle flags remain demo context, not login.

For a fresh checkout, create the environment first:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Test without calling the API

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

The tests use a fake Messages client. They verify the three-step handoff,
error result, conversation ordering, loop limit, and rejected out-of-scope
tool calls without API charges. SQLite tests create temporary databases and
check ownership, category and brand filters, parameterized queries, read-only
access, missing offers, and a complete agent loop. The suite currently has
131 offline tests, including repeated evaluation runs, follow-up history,
session isolation, per-turn authorization, failed-turn rollback, `/reset`,
tool-call and token budgets, sanitized logs, provider-error classification,
empty-catalog grounding checks, and shared source-contract checks for both
local adapters. `test_source_contract.py` checks each source through the same
application validator used at runtime. A future source can reuse the mixin
with its own fixed fixture IDs and application-owned label registry. It also
checks profile ownership and vehicle-specific catalog records.
The context tests also verify that malformed IDs are rejected before the
model request and that a conflicting ID written in the user's text cannot
authorize a different tool call.

## Small live evaluation

```powershell
.\.venv\Scripts\python.exe eval_agent.py
```

This runs twelve cases: general and paraphrased requests, ATE and Brembo brand
filters, a request for EUR when only RUB prices exist, an empty catalog, an
injected catalog failure, another user's vehicle, an unknown part number, a
price-source outage, a stale offer, and a prompt to bypass tool order. Exact checks verify
the expected tool sequence and arguments. In the empty and catalog-error
cases, the price tool must not be called.
For normal offers, the answer must include each returned part number and price,
identify mock data, and remind the user to verify by VIN. A separate Claude
Sonnet 5 call judges whether factual claims are supported by tool outputs.
Three calibration checks require the judge to reject a false brand claim, a
false “no matches” claim after a catalog error, and a false “invalid part”
claim after a price call was rejected before catalog search.

The command returns a nonzero exit code on any failure. Repeat `--case` to run
a subset, for example `--case eur_request --case empty_catalog`. Use
`--skip-calibration` after checking the judge once. To check stability on a
small subset, run:

```powershell
.\.venv\Scripts\python.exe eval_agent.py --case unknown_part --case bypass_order --repeat 3 --skip-calibration
```

`--repeat` accepts 1–10 runs per selected case (default 1), reports results
per case, and fails if any run fails. The three judge calibrations run once
per command unless skipped. API usage and cost grow with the number of case
runs, so start with a targeted subset. The expanded result is in
[`EVAL_EXPANDED.md`](EVAL_EXPANDED.md); the test matrix and gaps are in
[`EVAL_PLAN.md`](EVAL_PLAN.md), and the earlier two-case baseline is in
[`EVAL_BASELINE.md`](EVAL_BASELINE.md). Twelve cases still do not prove broad
reliability, and the LLM judge can make mistakes. Review failures manually and
expand the set before treating it as a release gate.

On 2026-09-23, two SQLite CLI smoke runs completed: vehicle 1 returned both
front-pad offers; vehicle 2 returned its front-pad part and correctly reported
no price offers. Two targeted mock-source regressions (front-pad paraphrase and
empty catalog) also passed. These are smoke checks, not a reliability estimate.

The conversation evaluation is separate from the single-turn set. By default
it runs the original three-turn brand refinement:

```powershell
.\.venv\Scripts\python.exe eval_multiturn.py --skip-calibration
```

It checks the initial front-pad request, a Brembo-only follow-up, and a price
re-check. Each turn must use fresh profile → catalog → price calls, preserve
the inferred category and brand, and ground its answer in that turn's data.
Three more named dialogues test a switch to an empty rear-disc catalog, a
catalog outage, and a price-source outage after a prior offer. Run one at a
time to control API cost, for example:

```powershell
.\.venv\Scripts\python.exe eval_multiturn.py --dialogue switch_category --skip-calibration
```

To sample whole-dialogue stability with a fresh session on every repetition:

```powershell
.\.venv\Scripts\python.exe eval_multiturn.py --dialogue catalog_outage --repeat 2 --skip-calibration
```

`--repeat` accepts 1–10 complete runs of each selected dialogue. It reports
per-dialogue and total pass counts; API calls and cost scale with turns and
repetitions. Source results are projected onto the validated vehicle, part,
and offer fields before the model sees them. Free-text notes and extra
metadata are excluded, including from evaluation traces. This is a narrow
prompt-injection boundary. Run the two focused allowed-field probes with:

```powershell
.\.venv\Scripts\python.exe eval_injection.py
```

These probes add harmless instruction text as extra raw `brand` or `seller`
fields while changing the corresponding ID to an unknown value. The application
must reject the unknown ID before the model sees the raw text. The probes check
`invalid_result`, correct tool stopping, and an answer without invented prices
or echoed attack text. Other source fields and unverified catalog facts remain
outside this narrow protection.

The dialogue judge receives previous tool results separately from current
results. It may use earlier results for explicitly historical claims, never
for current price or stock. `--calibrate-only` checks this distinction with
one valid and one invalid example. See [`EVAL_PLAN.md`](EVAL_PLAN.md) for the
test matrix, observed failures, and limitations.

## Current limitations

- Both sources contain fictional demo records, not verified vehicle
  compatibility or live offers. The SQLite file persists locally but is
  deliberately read-only to the agent.
- `observed_at` expresses when the demo price was captured or seeded, not
  provider authenticity. The 15-minute freshness window is a demonstration
  policy; a real provider needs a documented timestamp and update contract.
- The CLI's `--user-id` and `--vehicle-id` are demo context, not real
  authentication. A production application must derive these IDs from its
  authenticated session, never from the model or a user-supplied flag.
- A real provider must implement the three `CarDataSource` operations and
  supply trustworthy vehicle, catalog, and price data. The SQLite demo is
  not a real parts provider, and category normalization covers only a small
  English/Russian vocabulary.
- The current tool descriptions, system instructions, and RUB restriction still
  describe the mock. They must be reviewed before using a real provider.
- The schemas use Anthropic strict tool use. The per-run checks enforce a few
  business invariants, but do not replace complete input validation or
  authorization for a real database or commerce API.
- There are no retries, conversation persistence across restarts, RAG, or write
  actions yet. An in-memory session is not safe for concurrent requests from
  multiple processes or users; a production service needs authenticated,
  isolated session storage.
- The token threshold is a post-response circuit breaker, not a pre-call
  token count or exact cost guarantee. A future production service should also
  limit input size and track monetary usage against current provider pricing.
- A failed Anthropic request is classified as timeout, connection, limit,
  authentication, permission, service, or request error. The demo does not
  automatically retry these failures; a 429 can also indicate a spending cap,
  so blind retry would be inappropriate. The CLI reports a safe error code;
  a failed interactive turn leaves the prior conversation intact.
