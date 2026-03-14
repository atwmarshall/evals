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
systematic bias? This is an active research area — see Challenge 8 option A (reward hacking)
for a related failure mode.
