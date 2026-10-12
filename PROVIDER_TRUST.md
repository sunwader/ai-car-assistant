# Source IDs and provider trust boundary

Status: implemented for local demo data; external provider integration is a design, not a connected service.

## Requirements and assumptions

- Keep the three read-only tools and the Anthropic tool loop; no new agent framework.
- A source returns stable `brand_id` and `seller_id`, never model-ready display names.
- The application owns the ID → display-name registry. Unknown IDs fail with `invalid_result`.
- Existing name-based `demo.sqlite3` files remain readable without modifying them; offers without timestamps cannot support a current numeric price.
- The local mock and SQLite records are fictional. Neither proves VIN compatibility or live prices.
- A future provider must be chosen by application configuration, not by model-generated arguments.

## Data flow and contracts

```text
User request → model tool call → application authorization → source adapter
                                                     ↓
                                  part/offer records with stable IDs
                                                     ↓
                         result validation + application-owned ID registry
                                                     ↓
                             canonical names in model-visible tool_result
```

The source-side catalog record is `{"part_number": "P 59 045", "brand_id": "brand_brembo", "position": "front axle"}`. The model-visible record contains `"brand": "Brembo"`, resolved from `trusted_labels.py`; it does not receive a raw source `brand` field. Offers follow the same rule for `seller_id` → `seller`. Extra response fields are omitted. A requested brand name is resolved to an ID by the demo adapter before its catalog query; the application also verifies that every returned ID matches the requested brand.

New SQLite demo files store IDs in `parts.brand_id` and `offers.seller_id`. The SQLite adapter detects old name-based files read-only and translates recognized legacy names through the same registry. An unknown legacy name becomes an invalid result, not a new trusted label. The checked-in `demo.sqlite3` is not rewritten.

## Price observation and freshness (2026-09-24)

Assumption: `observed_at` is supplied by the source and means when that offer
was observed, not when the agent asked for it. The mock source stamps each
response at generation time; new SQLite fixtures stamp seeded offers when the
database is created. Neither is a live market feed. The application does not
replace a provider's old timestamp with its own current time.

```text
source offer + observed_at
    → source adapter
    → RunState: valid timezone + age ≤ 15 min + future skew ≤ 2 min
    ├─ pass → approved seller label + price in model-visible tool_result
    └─ fail → stale_price / invalid_result; no numeric offer exposed
```

An empty `offers` list still means a successful query with no offers. A
missing, malformed, or implausibly future timestamp yields `invalid_result`;
an observation older than 15 minutes yields `stale_price`. The whole price
result fails if any offer is stale, avoiding a mixture of fresh and stale
figures. The answer gate independently prevents a numeric RUB price without
a successful current-turn price result. Older SQLite files have no observation
column; the adapter reads them without migration and marks offer time unknown,
so the application refuses to quote those amounts.

The 15-minute window is a simple demo choice, not a provider SLA. A tighter
window reduces the chance of outdated quotes but rejects more legitimate
offers; a looser one does the reverse. Keeping this check in `RunState`
ensures mock, SQLite, and a future HTTP adapter share one decision. In a real
integration, set the threshold from the provider's documented update cadence,
verify the provider's timestamp semantics and clock behavior, and monitor
redacted counts of stale/invalid offers and observation ages. This check does
not authenticate the provider, prove part fitment, guarantee stock, or reserve
a price. Revisit the rule when a real provider contract and fixtures exist.

## How a real external provider should be authenticated

1. Configure one fixed HTTPS origin and API contract in the application. The model and user request must not supply the origin, arbitrary URL, credential, or headers.
2. Load a provider credential on the server from a secret manager or protected environment at startup. Give it read-only catalog/price scope, rotate it, and never put it in tool arguments, model context, traces, or ordinary logs.
3. Use an HTTP client with certificate verification enabled and explicit timeouts. Reject unexpected host changes and do not silently follow redirects to a new origin. Treat authentication, permission, timeout, malformed response, and provider outages as distinct failures.
4. Map provider entity IDs to application-owned brand/seller IDs through reviewed configuration or a governed mapping table. Never accept display names or an "approved" flag from the provider response as proof of trust. Unknown IDs fail closed and require review before mapping.
5. Verify the provider's vehicle/part relationship, currency, observation timestamp, and access rights independently of label resolution. The demo freshness check is not enough for commercial recommendations or ordering.
6. Keep user authentication separate from provider authentication. In production, derive `user_id` and selected `vehicle_id` from a server-side session, not the model or a CLI flag.

No real provider endpoint or credential is available in this project, so there is deliberately no HTTP adapter or credential-handling code yet.

## Executable source contract (2026-09-25)

`test_source_contract.py` provides one reusable set of checks for the two
local `CarDataSource` implementations. A future adapter can run those checks
against controlled provider fixtures by supplying a source, fixture user and
vehicle IDs, category, known brand/part, missing part, and an application-owned
`TrustedLabels` policy. The tests run source responses through the existing
`RunState` validator and require:

- a matching profile, a filtered catalog with the expected known part, and
  an observed price offer that passes the application contract;
- an unknown part to produce no invented offer, including when mixed with a
  known part;
- a mismatched user/vehicle profile request to fail, and catalog results to
  contain parts of the requested vehicle rather than another fixture vehicle;
- an unsupported currency to raise `DataSourceInvalidRequestError`, not to
  masquerade as a successful empty result.

This caught a demo-source gap: a direct request for an unknown part used to
create a fictitious 5,000 RUB offer. The mock now returns no offer for it.
The agent's catalog allowlist already blocked that direct request, but source
correctness is a separate defense. The conformance suite checks shape and
local behavior. The mock previously echoed arbitrary user/vehicle IDs and
returned Corsa parts for every vehicle; it now uses two explicit vehicle
fixtures and rejects unknown ownership pairs. Catalog and price methods do
not receive a user ID, so their direct calls are **not** an authorization
boundary: `RunState` limits them to the selected vehicle and its catalog
part numbers. The CLI's selected IDs are still not authenticated. These tests
cannot prove that a remote provider authenticated the customer, that its
catalog really fits a VIN, or that stock and prices are truthful. Those need
provider-specific fixtures, access-control tests, and independent
reconciliation before production use.

## Trade-offs and next review

The static tuple registry is tiny, immutable, dependency-free, and easy to audit. Its cost is manual onboarding for new brands and sellers; it is not suitable as an automatically growing production catalog. A governed mapping table can replace it when a real provider and update workflow exist, while preserving the same fail-closed ID contract. Before production, add contract tests against provider fixtures, credential/redirect tests, provider-specific freshness policy, redacted observability, and human review of newly mapped entities.
