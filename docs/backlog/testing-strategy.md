# Testing strategy

## Three distinct activities — don't conflate them

| Activity | What | When | Tool |
|---|---|---|---|
| Unit tests | Pure scorer functions | Write alongside each scorer | pytest |
| Smoke tests | End-to-end CLI runs | Ongoing | `uv run python -c "..."` |
| Dataset validation | Label quality, coverage | Before running benchmarks | Manual + sensitivity.py |

## Unit tests
Pure functions only — no API calls, no filesystem. Cheap, fast, run on every change.
Add to `tests/test_scorers.py` immediately when you write a scorer.
The contract to test: happy path, edge case, known failure mode.

## Smoke tests vs unit tests
The `uv run python -c "assert ..."` one-liners in plans are **smoke tests** — they
verify the code runs without errors but don't assert failure modes.
"Smoke tests pass" ≠ "unit tests pass". Know which you're claiming.

## Dataset validation is not testing
Asking "should I write tests for my dataset?" is a category error.
The dataset *is* the test. You don't test your tests — you *validate* them.
Validation means: manual review, label error checks, sensitivity analysis (`scripts/sensitivity.py`).

## What not to test
Runner and reporter integration tests require mocking the API client.
High boilerplate, low learning return. Skip for now.

## Rule
Pure function → write a unit test immediately.
Touches API or dataset → validate by running it and reading the output.
