# PTCG search agent

`search_agent.py` — for each MAIN decision (play/attach/evolve/attack/end),
uses the engine's own `search_begin`/`search_step` API to roll out each
candidate option a few steps forward (greedy policy for follow-up
sub-decisions on both sides, using a random determinization of hidden
information — our own deck/prize split and a symmetric-deck guess for the
opponent), then scores the resulting state with `heuristic.py` and picks the
best-scoring option. All other selection types fall back to a cheap
single-step heuristic rule.

Every real decision logs to stderr (`PTCG_DEBUG=1`, default on) — selection
type/context, option count, and for MAIN decisions the actual rollout step
count and per-option scores. This lets us verify directly in a real Kaggle
submission log that the search is actually executing, rather than silently
no-op'ing (the open question from the prior 760 Elo vs ~1034 Elo reference
gap).

`local_test.py` runs full battles locally against a random baseline using
the real compiled engine (`cg/libcg.dylib` on macOS) — no Kaggle round-trip
needed to sanity-check changes.

Local result: 19/20 wins vs random over alternating first-player seats.
