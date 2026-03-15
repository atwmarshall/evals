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
    │    ctx = ScorerContext(input=sample.input, metadata=sample.metadata)
    │    if isinstance(scorer, DatasetScorer):        ← dataset quality check
    │      completion = ""  (no model call)
    │      latency_ms = 0
    │      score = scorer(sample.expected, ctx)       ← 2-arg signature
    │    else:                                        ← model scorer
    │      completion = call_model(sample.input)
    │      score = scorer(completion, sample.expected, ctx)  # float | None
    │    yield RunResult(sample, completion, score, latency_ms, error)
    ▼
Reporter.report(results)
    │  prints table
    │  saves results/runs/{date}/{time}_{model}_{dataset}_{scorer}/run.json
    │                                                              samples.jsonl
    ▼
results/ directory
```

---

## The scorer contracts

There are two distinct scorer contracts. The type system in `core.py` encodes both:

```python
ScorerCallable      = Callable[[str, str, ScorerContext], float | None]  # model scorer
DatasetScorerCallable = Callable[[str, ScorerContext], float | None]     # dataset scorer
AnyScorer           = Union[ScorerCallable, DatasetScorerCallable]
```

**Model scorer** — evaluates model output:

```python
def scorer(completion: str, expected: str, ctx: ScorerContext) -> float | None:
    ...  # return 0.0 to 1.0, or None on parse/API failure
```

`ScorerContext` carries `input` (the original question) and `metadata` from the current `Sample`. Pure scorers accept it and ignore it. Context-aware scorers (LLMJudge, FaithfulnessScorer) use `ctx.input` as the question when prompting the judge.

Return `None` to signal a parse or API failure — the Runner records this as an error, not a score of 0.0.

**Dataset scorer** — evaluates dataset quality. Inherits from `DatasetScorer` (marker class). `completion` is structurally absent — it is not ignored, it does not appear in the signature:

```python
class MyDatasetScorer(DatasetScorer):
    def __call__(self, expected: str, ctx: ScorerContext) -> float | None:
        ...  # evaluate dataset quality from expected + ctx.metadata
```

Runner detects `isinstance(scorer, DatasetScorer)` and skips the model call. `RunResult.completion` is `""` and `latency_ms` is `0`. Run dataset scorers before model evals to validate dataset construction.

Class-based scorers (LLMJudgeScorer, CascadeScorer, FaithfulnessScorer, ContextSufficiencyScorer) hold state but their `__call__` satisfies one of the two contracts above.

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

`scripts/benchmark.py` loops the same dataset+scorer over multiple model IDs using the core `Runner`:

```python
for model_id in selected_models:
    config = EvalConfig(model=model_id)
    results = Runner().run(dataset, scorer, config)
    all_model_results.append((model_id, results))

reporter.benchmark_report(all_model_results, dataset_name, scorer_name)
```

The reporter writes `results/benchmarks/{date}/{time}_{dataset}_{scorer}/benchmark.json` plus one `{model_id}.jsonl` per model, then prints a comparison table (mean score, p50/p95 latency, error rate).

---

## LLM judge trace logging

Every call made by `LLMJudgeScorer` logs to `results/judge_traces/{date}/{time}_{model_id}/`:

```
results/judge_traces/2024-01-15/
└── 093000_llama3.2_3b/
    ├── c3-001.json    # {sample_id, evaluated_model, judge_prompt, raw_response, parsed_score, final_score, error}
    ├── c3-002.json
    └── ...
```

These are the most useful debugging artifact in the whole framework. Read them when scores seem wrong.

---

## Directory layout decisions

**Why `evals/scorers/` not `scorers/`?**  
Scorers are part of the eval library, not top-level scripts. They're imported by the runner.

**Why `scripts/benchmark.py` not `evals/benchmark.py`?**
The benchmark harness is a CLI script, not a library primitive. It uses the library but isn't part of it.

**Why JSONL not CSV or JSON array?**  
JSONL is streamable — you can iterate a million-sample dataset without loading it all into memory. It's also appendable (just cat lines). Both matter when datasets get large.

**Why `results/` not printed to stdout?**  
Results are structured data. They should be queryable, comparable across runs, and storable in version control. JSON files in `results/` give you a history of every eval run.
