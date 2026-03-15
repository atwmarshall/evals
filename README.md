# evals-from-scratch

A weekend project to build a working LLM evaluation framework from first principles.

**Goal**: understand how evals work — and why they're hard — by building every layer yourself.
**Stack**: Python 3.11+, Ollama, no eval frameworks.
**Companion**: see `docs/WEEKEND_PLAN.md` for the full challenge schedule.

---

## Project structure

```
evals-project/
├── evals/
│   ├── core.py              # Sample, Dataset, RunResult, ScorerContext, EvalConfig
│   ├── runner.py            # Calls the model, populates RunResult
│   ├── reporters.py         # Aggregates scores, prints tables, saves JSON
│   ├── scorer_factory.py    # build_scorer() — maps CLI args to scorer instances
│   └── scorers/
│       ├── exact.py         # exact_match, normalised_match
│       ├── regex.py         # RegexScorer, MultiRegexScorer
│       ├── schema.py        # JSONSchemaScorer
│       ├── llm_judge.py     # LLMJudgeScorer
│       ├── cascade.py       # CascadeScorer (fast + judge two-tier)
│       └── _json_utils.py   # _repair_truncated_json (internal)
├── runners/
│   └── benchmark.py         # BenchmarkRunner — runs one scorer across N models
├── scripts/
│   ├── run_eval.py          # CLI: run a single eval
│   ├── benchmark.py         # CLI: multi-model comparison
│   └── show.py              # CLI: inspect result artifacts
├── datasets/
│   ├── challenge2/          # Structured extraction samples (JSONL)
│   ├── challenge3/          # Open-ended QA samples (JSONL)
│   └── challenge7/          # RAG eval samples (JSONL)
├── tests/
├── results/                 # Auto-created — JSON/JSONL results from each run
├── CLAUDE.md                # Context for Claude Code
├── pyproject.toml
└── .env.example
```

---

## Quickstart

```bash
# 1. install deps
uv sync

# 2. configure Ollama
cp .env.example .env
# edit .env — set OLLAMA_HOST (default: http://localhost:11434)

# 3. run a single eval
uv run python scripts/run_eval.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer exact --model llama3.2:3b

# 4. run a multi-model benchmark
uv run python scripts/benchmark.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer schema --models llama3.2:3b,mistral:7b

# 5. inspect results
uv run python scripts/show.py results/runs/2024-01-01/123456_llama3.2_3b_extraction_exact/

# 6. run tests
uv run pytest
```

---

## Scorers

| Name | CLI flag | Returns | Notes |
|------|----------|---------|-------|
| `exact_match` | `--scorer exact` | `0.0` or `1.0` | Strict equality after strip |
| `normalised_match` | `--scorer normalised` | `0.0` or `1.0` | Lowercase, strip punctuation, collapse whitespace |
| `RegexScorer` | `--scorer regex --pattern "..."` | `0.0` or `1.0` | Pattern found in completion |
| `MultiRegexScorer` | `--scorer multi-regex --pattern "a" --pattern "b"` | `0.0`–`1.0` | Fraction of patterns matched |
| `JSONSchemaScorer` | `--scorer schema --schema path/to/schema.json` | `0.0`, `0.5`, or `1.0` | See partial credit below |
| `LLMJudgeScorer` | `--scorer judge` | `0.0`–`1.0` or `None` | Rubric-based, uses judge model |
| `CascadeScorer` | `--scorer cascade` | `0.0`–`1.0` or `None` | Fast tier first, judge as fallback |

**JSONSchemaScorer partial credit:**
- `1.0` — valid JSON that passes the schema
- `0.5` — valid JSON that fails the schema
- `0.0` — not valid JSON (after repair attempt)

**Score values:**
- `0.0`–`1.0` — normalised score (1.0 = correct)
- `None` — scorer failure (API error or unparseable judge response) — recorded as an error in results, distinct from a genuine low score

---

## Output fields

### Summary line (printed after each run)

| Field | Meaning |
|-------|---------|
| `mean_score` | Mean of all non-None scores. `—` if every sample errored. |
| `p50_latency` | Median latency across all samples (ms) |
| `p95_latency` | 95th-percentile latency (ms). Flagged with `⚠` when n < 20 (low confidence) |
| `api_errors` | Count of samples where the model API call failed entirely (no completion) |
| `parse_failures` | Count of samples where the scorer returned `None` (completion exists but scorer couldn't score it) |
| `error_rate` | `(api_errors + parse_failures) / n` as a percentage |
| `clean_rate` | % of samples where JSON parsed without any repair needed. Only shown for `schema` scorer. |
| `fmt_pass_rate` | % of samples where JSON was valid (clean or repaired). Only shown for `schema` scorer. |
| `repair_fail_rate` | % of samples where JSON could not be parsed or repaired. Only shown for `schema` scorer. |
| `judge_rate` | % of samples that fell through to the judge tier. Only shown for `cascade` scorer. |

### `scorer_metadata` per sample (in `samples.jsonl`)

Set by scorers to record per-sample diagnostic state. Keys present depend on which scorer was used.

| Key | Set by | Values |
|-----|--------|--------|
| `format_status` | `JSONSchemaScorer` | `"clean"` — parsed without repair<br>`"repaired"` — truncated JSON was fixed automatically<br>`"repair_failed"` — could not parse or repair |
| `judge_format_status` | `LLMJudgeScorer` | Same values as above, for the judge's own response. Absent on API error. |
| `tier_used` | `CascadeScorer` | `"fast"` — fast scorer met threshold, judge not called<br>`"judge"` — fast scorer missed, judge was called |
| `fast_score` | `CascadeScorer` | The raw score from the fast tier (`float` or `None`). Always present. |

### Result files

Each run saves to `results/runs/{date}/{time}_{model}_{dataset}_{scorer}/`:

| File | Contents |
|------|---------|
| `run.json` | Dataset, scorer, model, timestamp, summary stats |
| `samples.jsonl` | One JSON line per sample: `id`, `expected`, `score`, `latency_ms`, `completion`, `error`, `scorer_metadata` |

Each benchmark saves to `results/benchmarks/{date}/{time}_{dataset}_{scorer}/`:

| File | Contents |
|------|---------|
| `benchmark.json` | Per-model summary stats |
| `{model}.jsonl` | Same format as `samples.jsonl`, one file per model |

Judge traces (when using `judge` or `cascade` scorer) save to `results/judge_traces/{date}/{time}_{evaluated_model}/{sample_id}.json`. These are the most useful debugging artifact — they contain the full prompt, raw response, parsed score, and any error.

---

## show.py

Inspects result artifacts without opening raw files. Auto-detects the path type.

```bash
# Inspect a run
uv run python scripts/show.py results/runs/<date>/<run_dir>/

# Inspect a benchmark
uv run python scripts/show.py results/benchmarks/<date>/<bench_dir>/

# Show one sample by ID
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --id sample-001

# Show only failures (score < 1.0 or None)
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --failures-only

# Show full completions
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --verbose

# Inspect a raw JSONL file
uv run python scripts/show.py datasets/challenge2/extraction.jsonl

# Inspect judge traces
uv run python scripts/show.py results/judge_traces/<date>/<session_dir>/
```

---

## Dataset format

Each line in a JSONL dataset is:

```json
{"id": "001", "input": "...", "expected": "...", "metadata": {"category": "..."}}
```

- `input` — the prompt sent to the model
- `expected` — the target output (for deterministic scorers) or scoring rubric (for LLM judge)
- `metadata` — arbitrary key/value pairs passed through to `ScorerContext` and results

For LLM-judge datasets, `expected` is a rubric string describing what a good answer looks like, not a literal target answer.

---

## Key things to watch for

1. **Your scorer will be wrong.** Models produce correct answers that scorers mark as wrong (whitespace, capitalisation, phrasing). Understand *why* before making the scorer smarter.

2. **LLM judges have biases.** Position bias, verbosity bias, self-preference. You'll see these in Challenge 3.

3. **Benchmark rankings are task-dependent.** The model that wins on extraction loses on open-ended QA.

4. **Eval variance is real.** Small prompt changes can swing scores ±10%. Challenge 6 is dedicated to this.

5. **Dataset quality dominates everything.** A great scorer on a bad dataset gives false confidence. Challenge 5 is about breaking your dataset.

6. **`None` ≠ `0.0`.** A scorer returning `None` means it couldn't score the sample (API down, unparseable response). It's recorded separately from a genuine low score and excluded from mean calculation.
