# Latency metrics: p50, p95, p99

## What percentiles mean
Sort all response times fastest to slowest across N samples:
- **p50** — the middle value. 50% faster, 50% slower. The typical experience. Use this for model comparison.
- **p95** — 95% of calls were faster. What a user sees on a bad-but-not-worst-case request.
- **p99** — 99% of calls were faster. Your worst realistic case.

## Why not mean
Latency distributions are right-skewed. A few slow calls (timeouts, retries, cold starts)
drag the mean up significantly. Mean 800ms + p99 8000ms = "fine on average, terrible sometimes."
The mean hides this. p50 doesn't.

## Why p95/p99 matter at scale
At 1000 requests/minute, p99 happens 10 times per minute. "Rare" is still someone's
real experience.

## For this project
- Use **p50** as the primary comparison stat in the benchmark report
- Log **p95** alongside it — tells you whether a model is *consistently* faster
  or just *usually* faster with occasional spikes
- Those are different production characteristics and a benchmark that only reports p50 misses it

## Efficient vs efficient — be precise
"Most efficient cascade scorer" is ambiguous:
- **Computationally efficient** — minimise API calls (aggressive early stopping)
- **Statistically efficient** — maximise signal per sample (don't miss near-misses)

They pull in opposite directions. Know which you're optimising for before designing the cascade threshold.
