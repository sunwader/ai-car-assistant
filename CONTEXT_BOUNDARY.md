# Selected vehicle context boundary

Status: implemented for the local demo; user authentication is not implemented.

## Requirements and assumptions

- A model must not choose the user or selected vehicle for a run. A conflicting
  ID in a user request must not authorize a different profile or catalog.
- Demo CLI flags are application inputs, not credentials or proof of ownership.
- Keep the existing three read-only tools, in-memory sessions, and Anthropic
  Messages loop. Add no network request, database migration, or framework.
- Invalid or prompt-like IDs fail before a model request. Ordinary logs still
  omit the IDs and source results.

## Data flow

```text
demo CLI flags or future authenticated server session
    → SelectedVehicleContext (bounded plain IDs)
    → system instructions for the current run
    → RunState authorization of each model tool call
    → source adapter

raw user request → user message → model
```

`run_agent()` accepts the IDs as application arguments, constructs
`SelectedVehicleContext`, and keeps the user's text unchanged in its own
message. The context object rejects empty, long, whitespace-containing,
control-character, or punctuation-heavy IDs. The current demo permits ASCII
letters, digits, underscore, and hyphen, up to 64 characters. `AgentSession`
validates the same IDs and keeps one selection across its completed turns.
`RunState` still checks the exact IDs on every relevant tool call and result;
the system instruction is guidance, not the authorization mechanism.

No extra persistent store is needed for this local single-process demo. The
session history is in memory, and a failed turn does not commit history.

## Trade-offs and failure behavior

Separating application context from raw user text makes the model's instruction
hierarchy clearer. Validation keeps CLI-supplied IDs from becoming arbitrary
system-prompt text. Both are inexpensive and preserve existing tool schemas.
They do **not** authenticate the caller: anyone using the demo CLI can choose
a known user/vehicle pair. Nor can a Python value object stop an untrusted
backend from fabricating a context. If the model nevertheless requests a
different ID, `RunState` returns `invalid_request` without executing the
forbidden source call.

The fixed ID pattern is intentionally narrow for demo data. A real provider
may use other identifiers; revise the pattern only against its documented
format, keeping length and control-character limits. Do not accept free-form
user text or tool output as context. The model still sees selected IDs because
the current tool schemas require them; if that becomes undesirable, a later
change can bind identity inside the executor and remove ID fields from the
model-visible schemas.

## Production handoff and growth

A server must authenticate the user, resolve their selected vehicle, and check
ownership **before** calling `run_agent()`. It must derive `user_id` from its
verified session, not a query parameter, prompt, or model tool call. The
source adapter should enforce the same ownership relationship when loading a
profile; catalog and price calls should be scoped by the server's checked
selection, not merely a caller-supplied vehicle ID. Different users need
isolated session storage if the app becomes multi-process or multi-user.

For operations, record redacted counts of rejected context IDs and mismatched
tool calls without logging raw IDs or vehicle details. Revisit session storage,
ownership checks, ID format, and model-visible ID fields when an actual login
flow and parts-provider contract are chosen.
