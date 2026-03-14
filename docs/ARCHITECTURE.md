# Architecture

How the pieces fit together.

---

## Data flow

```
JSONL file
    │
    ▼
Dataset.from_jsonl()
    │  yields Sample(id, input, expected, metadata)
    ▼
Runner.run(dataset, scorer, config)
    │  for each sample:
    │    completion = call_model(sample.input)
    │    score = scorer(completion, sample.expected)
    │    yield RunResult(sample, completion, score, latency_ms, error)
    ▼
Reporter.report(results)
    │  prints table
    │  saves results/{timestamp}.json
    ▼
results/ directory
```

---

## The scorer contract

A scorer is any callable that satisfies:

```python
def scorer(completion: str, expected: str) -> float:
    ...  # return 0.0 to 1.0
```

That's it. Scorers are pure functions with no side effects. The one exception is `LLMJudge`, which is a class because it holds an API client — but its `score()` method still satisfies the contract.

This means you can compose scorers:

```python
def average_scorer(completion, expected):
    return (exact_match(completion, expected) + schema_scorer(completion, expected)) / 2
```

---

## RunResult

```python
@dataclass
class RunResult:
    sample: Sample
    completion: str | None    # None if error
    score: float | None       # None if error
    latency_ms: int
    error: str | None         # populated if API call failed
    metadata: dict            # scorer-specific data (judge reasoning, matched groups, etc.)
```

Errors are data. A failed API call produces a `RunResult` with `error` set — not an exception. The reporter counts error rates separately.

---

## Config

```python
@dataclass
class EvalConfig:
    model: str = "llama3.2"
    max_tokens: int = 1024
    temperature: float = 0.0
    system_prompt: str = ""
    max_retries: int = 3
    timeout_seconds: int = 30
```

Config is passed explicitly everywhere. No global config objects.

---

## Benchmark harness

`BenchmarkRunner` is a thin wrapper around `Runner` that loops over a list of model IDs:

```python
results_by_model = {}
for model_id in model_ids:
    config = EvalConfig(model=model_id)
    results_by_model[model_id] = runner.run(dataset, scorer, config)

reporter.benchmark_report(results_by_model)
```

The reporter then produces a comparison table.

---

## LLM judge trace logging

Every call made by `LLMJudge` logs to `results/judge_traces/{timestamp}/`:

```
results/judge_traces/2024-01-15T09-30-00/
├── trace-001.json    # {sample_id, judge_prompt, judge_response, score, reasoning}
├── trace-002.json
└── ...
```

These are the most useful debugging artifact in the whole framework. Read them when scores seem wrong.

---

## Directory layout decisions

**Why `evals/scorers/` not `scorers/`?**  
Scorers are part of the eval library, not top-level scripts. They're imported by the runner.

**Why `runners/benchmark.py` not `evals/benchmark.py`?**  
The benchmark harness is a runner script, not a library primitive. It uses the library but isn't part of it.

**Why JSONL not CSV or JSON array?**  
JSONL is streamable — you can iterate a million-sample dataset without loading it all into memory. It's also appendable (just cat lines). Both matter when datasets get large.

**Why `results/` not printed to stdout?**  
Results are structured data. They should be queryable, comparable across runs, and storable in version control. JSON files in `results/` give you a history of every eval run.
