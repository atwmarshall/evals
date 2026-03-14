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
│   ├── core.py              # Dataset, Sample, RunResult — the primitives
│   ├── runner.py            # Calls the model, stores completions
│   ├── reporters.py         # Aggregates scores, prints tables, saves JSON
│   └── scorers/
│       ├── __init__.py
│       ├── exact.py         # Exact match, normalised match
│       ├── regex_scorer.py  # Regex-based scoring
│       ├── schema.py        # JSON schema validation scorer
│       └── llm_judge.py     # LLM-as-judge scorer (Challenge 3)
├── runners/
│   └── benchmark.py         # Multi-model benchmark harness (Challenge 4)
├── datasets/
│   ├── challenge2/          # Structured extraction samples (JSONL)
│   ├── challenge3/          # Open-ended QA samples (JSONL)
│   └── challenge7/          # RAG eval samples (JSONL)
├── tests/
│   └── test_scorers.py      # Unit tests for each scorer
├── scripts/
│   ├── run_eval.py          # CLI entrypoint: run any eval suite
│   └── sensitivity.py       # Sensitivity analysis (Challenge 6)
├── docs/
│   ├── WEEKEND_PLAN.md      # Full challenge schedule + goals
│   ├── ARCHITECTURE.md      # How the framework fits together
│   ├── FAILURE_MODES.md     # Living doc — fill this in as you go
│   └── OPEN_PROBLEMS.md     # Known hard problems in evals (Challenge 8)
├── results/                 # Auto-created — JSON results from each run
├── CLAUDE.md                # Context for Claude Code
├── pyproject.toml           # Dependencies (managed with uv)
└── .env.example
```

---

## Quickstart

```bash
# 1. clone / enter the project
cd evals-project

# 2. install deps (uv creates the venv automatically)
uv sync

# 3. configure Ollama connection
cp .env.example .env
# edit .env and set OLLAMA_HOST (default: http://localhost:11434)

# 4. run the first eval
uv run python scripts/run_eval.py --dataset datasets/challenge2/extraction.jsonl --scorer exact

# 5. run the benchmark harness (Challenge 4)
uv run python scripts/run_eval.py --dataset datasets/challenge2/extraction.jsonl \
  --scorer schema --models llama3.2 mistral
```

---

## The four primitives

Everything in this framework is built from four classes:

| Class | File | Responsibility |
|---|---|---|
| `Sample` | `evals/core.py` | One `(input, expected)` pair |
| `Dataset` | `evals/core.py` | Loads + iterates samples from JSONL |
| `Runner` | `evals/runner.py` | Calls the model, returns `RunResult` |
| `Reporter` | `evals/reporters.py` | Aggregates, prints, saves |

A scorer is just a function: `(completion: str, expected: str) -> float`.

---

## Running a challenge

Each challenge has its own section in `docs/WEEKEND_PLAN.md` with goals, what to watch for, and prompts to try. The challenges build on each other — don't skip ahead.

---

## Key things to watch for (read before you start)

1. **Your scorer will be wrong.** The first thing you'll notice is that models produce correct answers your scorer marks as wrong (whitespace, capitalisation, phrasing). Resist the urge to make the scorer smarter immediately — first understand *why* it's strict.

2. **LLM judges have biases.** Position bias (order of options matters), verbosity bias (longer = better), self-preference (Claude prefers Claude). You'll see these in Challenge 3.

3. **Benchmark rankings are task-dependent.** The model that wins on extraction loses on open-ended. Never trust a single number.

4. **Eval variance is real.** Small prompt changes can swing scores ±10%. This is the most underappreciated problem in the field. Challenge 6 is dedicated to it.

5. **Dataset quality dominates everything.** A great scorer on a bad dataset gives you false confidence. Challenge 5 is dedicated to breaking your dataset.

---

## Filling in `docs/FAILURE_MODES.md`

Every time something surprises you — a score that seems wrong, a model that behaves unexpectedly, a scorer that misfires — add it to `docs/FAILURE_MODES.md`. By Sunday evening this document is the most valuable output of the weekend.
