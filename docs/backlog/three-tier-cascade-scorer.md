# Cascade scorer

## What it is
A chain of scorers where each tier either returns a result or passes to the next.
Not recursive — it's a pipeline. Each tier terminates or escalates.

## Current implementation
```
normalised_match → LLM judge
```
Fast heuristic gates the expensive call. Returns 1.0 immediately on match.

## Known limitation
Fast tier is hardcoded. Better design accepts any scorer as the fast tier:
```python
CascadeScorer(fast=exact_match, judge=my_judge)
```

## Next: three-tier model cascade
```
fast heuristic → small LLM → large LLM
```
Middle tier only fires when heuristic fails. Large model only fires when small model is uncertain.

## Two approaches to "not confident enough"

**Score threshold** — escalate when score falls in uncertainty band:
```python
if 0.4 <= score <= 0.8:
    return self._large_judge(completion, expected, ctx)
```

**Explicit confidence field** — ask the model to return `{"score": 3, "confidence": "low"}`.
More reliable — model reasons about its own uncertainty rather than you inferring it from score value.

## Why it matters
At scale, p99 of requests hitting the large model is expensive. A well-calibrated
small model can handle 70–80% of cases. The economics are significant.

## Open question
Are small models reliable enough to gate large model calls without introducing
systematic bias? This is an active research area — see OPEN_PROBLEMS.md (reward hacking)
for a related failure mode.

## Observability: tier_used and judge_rate

`CascadeScorer` writes two keys to `ctx.metadata_out` on every call:
- `tier_used`: `"fast"` if the fast scorer met the threshold, `"judge"` if the judge was called
- `fast_score`: the raw score from the fast tier before the escalation decision

`Reporter._summarise()` derives `judge_rate` — the fraction of samples that escalated to the
judge. This appears in `benchmark.json` and the benchmark comparison table so you can see how
often the fast tier is saving judge calls across models.
