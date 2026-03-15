# Metric decomposition

## The problem

A score of 0.0 from `JSONSchemaScorer` is ambiguous: did the model produce the wrong answer,
or did it fail to produce valid JSON at all? The distinction matters — one is a capability
failure, the other is a format failure. Lumping them into `mean_score` hides the signal.

Similarly, `LLMJudgeScorer` returning `None` (parse failure) is already separated from a
genuine 0.0, but *why* the parse failed (bad JSON vs API error) was invisible.

## Solution: `metadata_out` side-channel

Rather than changing the scorer return type (`float | None` → `tuple[float | None, dict]`),
scorers write diagnostic metadata to `ctx.metadata_out`. Runner copies it to
`RunResult.metadata` after the call.

This is non-breaking: pure scorers (exact_match, regex, normalised) don't touch `metadata_out`
and require zero changes. Only scorers that have something to say need to write to it.

```python
# ScorerContext (evals/core.py)
metadata_out: dict = field(default_factory=dict)

# Runner pattern
ctx = ScorerContext(input=..., metadata=..., metadata_out={})
score = scorer(completion, expected, ctx)
RunResult(..., metadata=ctx.metadata_out)
```

## Standardised metadata_out keys

| Key | Set by | Values | Meaning |
|-----|--------|--------|---------|
| `format_status` | `JSONSchemaScorer` | `"clean"` / `"repaired"` / `"repair_failed"` | Whether the JSON parsed cleanly, needed repair, or was unparseable |
| `judge_format_status` | `LLMJudgeScorer` | `"clean"` / `"repaired"` / `"repair_failed"` | Same for the judge's own response |
| `tier_used` | `CascadeScorer` | `"fast"` / `"judge"` | Which tier produced the final score |
| `fast_score` | `CascadeScorer` | `float \| None` | Fast tier score before escalation decision |

## New summary metrics

`Reporter._summarise()` computes these from `r.metadata` across all results:

- `clean_rate` — fraction of samples with `format_status == "clean"` (None if scorer doesn't set it)
- `format_pass_rate` — fraction with `format_status in ("clean", "repaired")` (parseable JSON)
- `repair_failure_rate` — fraction with `format_status == "repair_failed"` (unparseable)
- `judge_rate` — fraction of samples where `tier_used == "judge"` (None if not CascadeScorer)

These appear in `benchmark.json` and the benchmark comparison table when non-None.
`samples.jsonl` includes `scorer_metadata` per row so `show.py --id` can display them.

## Why not change the return type?

Changing `(str, str, ScorerContext) -> float | None` to return a tuple would require
updating runner.py, cascade.py, all 5 scorer files, all scorer tests, and all call sites.
The side-channel approach costs zero changes to existing code and is opt-in.
