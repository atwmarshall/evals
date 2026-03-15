# evals-from-scratch

A weekend project to build a working LLM evaluation framework from first principles.

**Goal**: understand how evals work — and why they're hard — by building every layer yourself.
**Stack**: Python 3.11+, Ollama, no eval frameworks.
**Companion**: see `docs/WEEKEND_PLAN.md` for the full schedule.

---

## Project structure

```
evals/
├── evals/
│   ├── core.py                    # Sample, Dataset, RunResult, ScorerContext, EvalConfig, ScorerCallable
│   ├── runner.py                  # Calls the model, populates RunResult
│   ├── reporters.py               # Aggregates scores, prints tables, saves JSON
│   ├── scorer_factory.py          # build_scorer() — maps CLI args to scorer instances
│   ├── sensitivity_reporter.py    # run_variations(), SensitivityReporter
│   ├── variation_generator.py     # VariationGenerator — LLM-based input rewriting
│   └── scorers/
│       ├── exact.py               # exact_match, normalised_match
│       ├── regex.py               # RegexScorer, MultiRegexScorer
│       ├── schema.py              # JSONSchemaScorer
│       ├── llm_judge.py           # LLMJudgeScorer
│       ├── cascade.py             # CascadeScorer (fast + judge two-tier)
│       └── _json_utils.py         # _repair_truncated_json (internal)
├── scripts/
│   ├── run_eval.py                # CLI: run a single eval
│   ├── benchmark.py               # CLI: multi-model comparison
│   ├── sensitivity.py             # CLI: scorer sensitivity analysis
│   └── show.py                    # CLI: inspect result artifacts
├── datasets/
│   ├── challenge2/                # Structured extraction samples (JSONL)
│   ├── challenge3/                # Open-ended QA samples (JSONL)
│   ├── challenge7/                # RAG eval samples (JSONL)
│   └── generated/
│       └── sensitivity/           # Auto-created by sensitivity.py
├── tests/
├── results/                       # Auto-created
│   ├── runs/
│   ├── benchmarks/
│   ├── sensitivity/
│   └── judge_traces/
├── CLAUDE.md
├── pyproject.toml
└── .env.example
```

---

## Setup

```bash
uv sync
cp .env.example .env
# edit .env — set OLLAMA_HOST and model vars (see Environment variables below)
```

---

## Scripts

### `run_eval.py` — single eval run

```bash
uv run python scripts/run_eval.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer exact --model llama3.2:3b

# Schema scorer with a JSON schema file
uv run python scripts/run_eval.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer schema --schema datasets/challenge2/schema.json

# LLM judge, limit to 20 samples
uv run python scripts/run_eval.py \
  --dataset datasets/challenge3/qa.jsonl \
  --scorer judge --limit 20

# Cascade: normalised match first, judge fallback
uv run python scripts/run_eval.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer cascade --fast-tier normalised
```

Flags: `--dataset`, `--scorer`, `--model`, `--limit`, `--output`, plus scorer-specific flags (see Scorers table).

Results saved to `results/runs/{date}/{time}_{model}_{dataset}_{scorer}/`.

---

### `benchmark.py` — multi-model comparison

Runs the same scorer across multiple models and prints a comparison table (mean score, p50/p95 latency, error rate).

```bash
# Pass models directly
uv run python scripts/benchmark.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer schema --schema datasets/challenge2/schema.json \
  --models llama3.2:3b,mistral:7b

# Interactive model selection (questionary checkbox)
uv run python scripts/benchmark.py \
  --dataset datasets/challenge3/qa.jsonl \
  --scorer judge
```

Flags: `--dataset`, `--scorer`, `--models` (comma-separated, skips interactive), plus scorer-specific flags.

Results saved to `results/benchmarks/{date}/{time}_{dataset}_{scorer}/`.

---

### `sensitivity.py` — scorer sensitivity analysis

Holds the model fixed and varies input phrasing across 5 variation types. Measures score variance to test scorer reliability: **"can I trust my ruler?"**

#### The pipeline

```
load dataset
  → VariationGenerator.generate()      # 5 variation types via LLM
  → validate_variations()              # LLM judge filters out meaning-changed samples
  → run_variations()                   # Runner through eval scorer for each variation
  → SensitivityReporter.report()       # variance tables + sensitivity.json
```

#### Three-model requirement

Sensitivity analysis uses three distinct models — this is intentional and enforced:

| Role | Env var | Description |
|------|---------|-------------|
| Evaluated model | `DEFAULT_MODEL` | The model whose completions are scored |
| Judge | `JUDGE_MODEL` | Scores completions via `LLMJudgeScorer`; also used for variation validation |
| Variation generator | `VARIATION_MODEL` | Rewrites inputs via `VariationGenerator` |

**`VARIATION_MODEL` must differ from `JUDGE_MODEL`.** Using the same model to generate and score variations measures self-consistency, not scorer reliability — the script will exit with an error if they match.

#### Variation types

| Type | What it does |
|------|-------------|
| `synonym_swap` | Replaces key words with synonyms |
| `rephrase` | Rewrites the sentence structure |
| `add_noise` | Adds minor phrasing noise |
| `formal` | Makes the phrasing more formal |
| `concise` | Shortens the input |

#### Validation

Before running the scorer, each variation is checked by the judge for meaning preservation. Variations that score below `--validation-threshold` (default `0.8`) are dropped. Filtering is per-sample: if a variation fails for one sample, only that sample is dropped for that variation type.

#### Reusing saved variations

By default, variations are saved to `datasets/generated/sensitivity/` after generation and validation. To re-run analysis with a different scorer on the same inputs, skip generation entirely:

```bash
--reuse-variations datasets/generated/sensitivity/2026-03-15_extraction_llama3.2:3b/
```

This loads validated variations from disk, so variation generation and validation are skipped.

#### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | required | Path to JSONL dataset |
| `--scorer` | required | Scorer to test for reliability |
| `--model` | `DEFAULT_MODEL` | Model being evaluated |
| `--limit` | none | Max samples |
| `--variations` | all 5 | Space-separated list of variation types to run |
| `--validation-threshold` | `0.8` | Min judge score for meaning preservation |
| `--no-save-variations` | off | Skip saving generated variations |
| `--reuse-variations DIR` | none | Load variations from a saved directory |
| `--output` | `RESULTS_DIR` | Override output directory |
| `--judge-model` | `JUDGE_MODEL` | Override judge model for scoring and validation |
| `--scale` | `5` | Judge score scale |
| `--fast-tier` | `normalised` | Fast tier for cascade scorer |
| `--cascade-threshold` | `1.0` | Fast-tier threshold (separate from `--validation-threshold`) |

#### Output

Per-sample variance table and per-variation summary table are printed to stdout. Verdict is `unstable` if variance > 0.05, `ok` otherwise.

Results saved to `results/sensitivity/{date}/{time}_{dataset}_{scorer}/`.

#### Examples

```bash
# Full run — generates and saves variations
uv run python scripts/sensitivity.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer judge --limit 10

# Reuse previously saved variations (different scorer)
uv run python scripts/sensitivity.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer cascade \
  --reuse-variations datasets/generated/sensitivity/2026-03-15_extraction_llama3.2:3b/

# Run only specific variation types
uv run python scripts/sensitivity.py \
  --dataset datasets/challenge2/extraction.jsonl \
  --scorer judge --variations rephrase formal
```

---

### `show.py` — inspect result artifacts

Auto-detects the path type in this order: sensitivity dir → benchmark dir → run dir → JSONL → traces dir.

```bash
# Inspect a run
uv run python scripts/show.py results/runs/<date>/<run_dir>/

# Inspect a benchmark
uv run python scripts/show.py results/benchmarks/<date>/<bench_dir>/

# Inspect a sensitivity result
uv run python scripts/show.py results/sensitivity/<date>/<sens_dir>/

# Show only unstable samples (sensitivity mode)
uv run python scripts/show.py results/sensitivity/<date>/<sens_dir>/ --failures-only

# Show one sample by ID
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --id sample-001

# Show full completions
uv run python scripts/show.py results/runs/<date>/<run_dir>/ --verbose

# Inspect a raw JSONL file
uv run python scripts/show.py datasets/challenge2/extraction.jsonl

# Inspect judge traces
uv run python scripts/show.py results/judge_traces/<date>/<session_dir>/
```

In sensitivity mode, `--failures-only` filters the per-sample table to rows where `verdict == "unstable"`.

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

In `sensitivity.py`, the scorer is tested for reliability — the same scorer runs across all variation types to measure whether it produces consistent scores.

### Scorer-specific flags

| Flag | Scorer | Default |
|------|--------|---------|
| `--pattern` | `regex`, `multi-regex` | required |
| `--schema` | `schema` | required |
| `--scale` | `judge`, `cascade` | `5` |
| `--judge-model` | `judge`, `cascade` | `JUDGE_MODEL` env |
| `--fast-tier` | `cascade` | `normalised` |
| `--cascade-threshold` | `cascade` (in `sensitivity.py`) | `1.0` |

---

## Results directory structure

```
results/
├── runs/{date}/{time}_{model}_{dataset}_{scorer}/
│   ├── run.json
│   └── samples.jsonl
├── benchmarks/{date}/{time}_{dataset}_{scorer}/
│   ├── benchmark.json
│   └── {model}.jsonl
├── sensitivity/{date}/{time}_{dataset}_{scorer}/
│   ├── sensitivity.json
│   ├── baseline.jsonl
│   ├── rephrase.jsonl
│   └── ...
└── judge_traces/{date}/{time}_{model}/
    └── {sample_id}.json
```

### `sensitivity.json` fields

| Field | Contents |
|-------|---------|
| `dataset` | Source dataset name |
| `scorer` | Scorer used for reliability testing |
| `model` | Evaluated model |
| `timestamp` | ISO timestamp |
| `materialised_at` | When the file was written |
| `run_config` | `variation_model`, `judge_model`, `validation_threshold`, `cascade_threshold`, `limit`, `reused_from` |
| `variation_names` | List of variation types that were run (including `"baseline"`) |
| `schema_notes` | Internal schema versioning notes |
| `per_sample` | List of per-sample rows: `id`, one score per variation, `variance`, `verdict` |
| `per_variation` | List of per-variation rows: `variation`, `mean_score`, `delta_from_baseline`, `mean_variance` |
| `summary` | Top-level summary: `n_unstable`, `n_total`, `most_destabilising` |

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

For RAG datasets, samples include an additional `context` field:

```json
{"id": "rag-01", "input": "What is the capital of France?", "expected": "Paris", "context": ["France is a country in Western Europe. Its capital is Paris."], "metadata": {"category": "factual"}}
```

---

## Environment variables

| Variable | Used by | Default | Notes |
|----------|---------|---------|-------|
| `OLLAMA_HOST` | Runner, all scorers | `http://localhost:11434` | |
| `DEFAULT_MODEL` | `EvalConfig` | `llama3.2:3b` | Model being evaluated |
| `JUDGE_MODEL` | `LLMJudgeScorer`, `CascadeScorer` | `llama3.2:3b` | Scores completions |
| `VARIATION_MODEL` | `VariationGenerator` | falls back to `DEFAULT_MODEL` | Must differ from `JUDGE_MODEL` in sensitivity analysis |
| `MAX_TOKENS` | `EvalConfig` | `1024` | |
| `RESULTS_DIR` | `Reporter`, `SensitivityReporter` | `results/` | |

---

## Key things to watch for

1. **Your scorer will be wrong.** Models produce correct answers that scorers mark as wrong (whitespace, capitalisation, phrasing). Understand *why* before making the scorer smarter.

2. **LLM judges have biases.** Position bias, verbosity bias, self-preference. You'll see these clearly with the judge scorer.

3. **Benchmark rankings are task-dependent.** The model that wins on extraction loses on open-ended QA.

4. **Eval variance is real.** Small prompt changes can swing scores ±10%. Sensitivity analysis is dedicated to measuring this.

5. **Dataset quality dominates everything.** A great scorer on a bad dataset gives false confidence.

6. **`None` ≠ `0.0`.** A scorer returning `None` means it couldn't score the sample (API down, unparseable response). It's recorded separately from a genuine low score and excluded from mean calculation.

7. **Three-model separation matters in sensitivity analysis.** If `VARIATION_MODEL` equals `JUDGE_MODEL`, you're measuring self-consistency, not scorer reliability. The script enforces separation and exits on misconfiguration.
