# Weekend plan — evals from scratch

8 challenges across 2 days. Each builds on the last. The system you build is the same system throughout — you're adding layers, not starting over.

---

## Saturday — build the machine

### Challenge 1 · 9–11am · The harness core

**Goal**: a working eval pipeline that can run a dataset through a model and report a score.

**Build**:
- `evals/core.py` — `Sample`, `Dataset`, `RunResult`, `EvalConfig` dataclasses
- `evals/runner.py` — `Runner.run(dataset, scorer, config) -> list[RunResult]`
- `evals/reporters.py` — `Reporter.report(results)` prints table + saves JSON

**Start here**: write `Dataset.from_jsonl()` first. Force yourself to decide: what fields does a sample need? You'll immediately hit the question of what `expected_output` means. Write it down.

**The wall you'll hit**: there's no single right format for `expected`. For extraction tasks it might be a string. For open-ended tasks it might be a rubric. For code it might be a test suite. Design `Sample` to handle all of these — use a `metadata` dict for task-specific fields.

**You've finished when**: you can run `python scripts/run_eval.py --dataset datasets/challenge2/extraction.jsonl --scorer exact` and see a results table.

**What to notice**: how long does the API take per sample? What happens when a call fails? Where do you put retry logic?

---

### Challenge 2 · 11am–1pm · Deterministic scorers

**Goal**: implement exact match, regex, and JSON schema scoring. Evaluate a structured extraction task. Experience the tension between strictness and coverage.

**Build**:
- `evals/scorers/exact.py` — `exact_match`, `normalised_match`
- `evals/scorers/regex_scorer.py` — `RegexScorer`, `MultiRegexScorer`
- `evals/scorers/schema.py` — `JSONSchemaScorer` with partial credit
- `datasets/challenge2/extraction.jsonl` — 20+ samples: extract name, date, amount from short texts

**Sample dataset entries to create**:
```jsonl
{"id": "c2-001", "input": "Invoice from Acme Corp dated 15 Jan 2024 for £1,250.00", "expected": "{\"company\": \"Acme Corp\", \"date\": \"2024-01-15\", \"amount\": 1250.00}", "metadata": {"category": "invoice_extraction"}}
{"id": "c2-002", "input": "Meeting with Sarah Chen on March 3rd re: Q1 budget", "expected": "{\"person\": \"Sarah Chen\", \"date\": \"2024-03-03\", \"topic\": \"Q1 budget\"}", "metadata": {"category": "meeting_extraction"}}
```

**The wall you'll hit**: your `exact_match` scorer will mark correct answers wrong. The model says `"January 15, 2024"` and your expected is `"2024-01-15"`. You'll want to fix the scorer. Don't — yet. First: is this a scorer problem or a prompt problem? Try fixing the prompt to get the model to output your exact format. Now you understand why format constraints in prompts matter.

**Then**: implement `normalised_match` and see how much it helps. Then implement `JSONSchemaScorer` — suddenly partial credit is possible, and your score distribution gets much more interesting.

**What to measure**: pass rate for each scorer. How much does normalisation help? What's the gap between exact and schema scoring?

---

### Challenge 3 · 2–4pm · LLM-as-judge

**Goal**: build a grader that uses Claude to evaluate open-ended answers. Then deliberately break it to find its failure modes.

**Build**:
- `evals/scorers/llm_judge.py` — `LLMJudge(criteria, scale=5)`
- `datasets/challenge3/openended.jsonl` — 15+ open-ended QA samples
- `results/judge_traces/` — the judge logs every prompt + response here

**The judge prompt template**:
```
You are an expert evaluator. Score the following answer on a scale of 1 to {scale}.

Question: {question}
Answer: {answer}

Criteria: {criteria}

Respond with JSON only:
{"score": <int 1-{scale}>, "reasoning": "<one sentence>"}
```

**The experiments to run** (this is the heart of Challenge 3):

1. **Position bias**: create pairs where you swap the order in which you present two answers. Does the judge prefer whichever comes first?
2. **Verbosity bias**: take a correct short answer and pad it with irrelevant sentences. Does the score go up?
3. **Self-preference**: compare Claude judging Claude vs Claude judging GPT-4o outputs (use saved completions). Does Claude prefer itself?
4. **Criteria sensitivity**: change "be concise" to "be thorough". How much do scores shift on the same answers?

**What to notice**: the judge traces are more informative than the scores. Read them. The reasoning reveals what the judge is actually paying attention to — it's often not what you told it to evaluate.

**You've finished when**: you can name three specific biases you observed in your judge, with examples.

---

### Challenge 4 · 4–6pm · Benchmark harness

**Goal**: run your full suite across multiple models. Build a comparison table. Discover that model rankings are task-dependent.

**Build**:
- `runners/benchmark.py` — `BenchmarkRunner` takes a list of model IDs
- Extend `reporters.py` to output a comparison table

**Models to compare** (use at least two):
- `claude-haiku-4-5-20251001` — fast, cheap, good baseline
- `claude-sonnet-4-6` — stronger, slower
- Optionally add a GPT-4o-mini or Gemini Flash via their APIs

**The comparison table should show**:
```
Model                    | Extraction (schema) | Open-ended (judge) | Mean latency | Error rate
-------------------------|--------------------|--------------------|--------------|----------
claude-haiku-4-5-20251001|       0.71         |        3.2/5       |    420ms     |   0%
claude-sonnet-4-6        |       0.84         |        4.1/5       |    890ms     |   0%
```

**What to notice**: Sonnet will likely win overall, but by how much? Is the gap worth the cost and latency? On which tasks does Haiku surprise you? Are there cases where Haiku beats Sonnet?

**The deeper question**: if you had to ship one model for production, which metrics actually matter for your use case? Cost/score tradeoff is a real engineering decision.

---

### Challenge 5 (end of Saturday) · 6–7pm · Retro

Write exactly five bullet points in `docs/FAILURE_MODES.md`:
- One thing your scorer got wrong that the model got right
- One thing your judge was obviously biased about
- One thing you didn't anticipate when designing `Sample`
- One model behaviour that surprised you
- One thing you'd design differently from scratch

These are prompts for Sunday.

---

## Sunday — stress-test the machine

### Challenge 5 · 9–11am · Dataset curation

**Goal**: make your dataset harder. Discover that most of your Saturday samples were too clean.

**Three things to add to your datasets**:

1. **Adversarial inputs** — inputs designed to confuse *the scorer*, not the model:
   - Answers that are correct but phrased in a way your exact-match scorer misses
   - JSON with extra fields your schema scorer doesn't expect
   - Answers that match your regex pattern but are semantically wrong

2. **Distribution gaps** — inputs the model hasn't "seen" in your existing set:
   - Different domains (your extraction dataset was invoices — add medical records, legal contracts)
   - Different languages (try a non-English input)
   - Unusual formatting (tabular input, markdown, code comments)

3. **Ambiguous cases** — inputs where the "correct" answer is genuinely debatable:
   - Questions with multiple valid answers
   - Extraction tasks where the source text is ambiguous
   - Factual questions that are contested or time-sensitive

**What to measure**: how much does your overall score drop when you add these? What's the breakdown by category? The categories that drop most are where your eval is weakest — and likely where your model is weakest too.

**The core insight**: your Saturday dataset was a *sample* of easy cases. Real-world inputs are not a random sample — they're skewed toward the hard cases that reach your system. A good eval dataset is adversarially curated, not randomly sampled.

---

### Challenge 6 · 11am–1pm · Sensitivity analysis

**Goal**: find out how stable your eval scores are. Small input changes should produce small score changes. If they don't, your eval has low reliability.

**Build**: `scripts/sensitivity.py`

**The experiment**:
Take 10 samples from your dataset. For each, create 5 prompt variations:
- Rephrase the instruction (same meaning, different words)
- Add or remove a trailing period
- Change "extract" to "identify" or "find"
- Change example order in few-shot prompts
- Add an irrelevant sentence to the input

Run each variation through your runner. Record the score for each. Compute variance across variations.

**What to measure**:
- Mean score variance per sample
- Which scorer is most sensitive? (Exact match, schema, or LLM judge?)
- Which sample categories are most sensitive?

**The wall you'll hit**: your LLM judge will show the highest variance. A prompt change that doesn't change the meaning at all will shift scores by 0.5–1.0 points on a 5-point scale. This is a known, serious problem — it means your eval scores have a noise floor that makes small improvements invisible.

**What this means practically**: if you're using evals to guide prompt engineering, you need to run each variant multiple times and average — not trust a single score.

---

### Challenge 7 · 2–4pm · RAG eval suite

**Goal**: apply your framework to a real use case. Build a small RAG system and evaluate it across three dimensions.

**Build**:
1. A tiny RAG pipeline — 10–20 documents, keyword or embedding retrieval, Claude for answer generation
2. Three scorers for RAG:
   - **Retrieval relevance** — did the retrieved chunks actually contain the answer? (regex or schema scorer)
   - **Faithfulness** — does the answer contradict the retrieved context? (LLM judge with a specific faithfulness rubric)
   - **Answer quality** — is the answer good? (LLM judge with a quality rubric)
3. A dataset of 15+ questions answerable from your documents

**The interesting tension**: faithfulness and quality conflict. A faithful answer that stays close to the context is often worse than a slightly hallucinated but more helpful one. Your eval has to decide which it values more — and that decision encodes your product values.

**Faithfulness judge prompt**:
```
You are evaluating whether an answer is faithful to the provided context.
An answer is faithful if every claim in it is supported by the context.
An answer is unfaithful if it introduces facts not in the context, even if those facts are true.

Context: {context}
Answer: {answer}

Score: 1 (completely unfaithful) to 5 (completely faithful)
```

**What to measure**: the correlation between retrieval relevance and answer quality. When retrieval fails, does quality always fail too? (Probably not — the model will hallucinate plausibly, which is the scary case.)

---

### Challenge 8 · 4–6pm · Open problem deep dive

**Pick one**. Go as deep as you can in two hours.

---

**Option A: Reward hacking**

Build an eval, then try to make the model game it without actually improving.

How: write a system prompt that tells the model "you will be evaluated by [your exact scorer logic]. Optimise for score, not for quality." Run it. Compare scores vs quality (have your LLM judge score the reward-hacked outputs independently). The gap between eval score and actual quality is reward hacking.

This is the central problem of AI alignment. Your weekend-scale version of it is real.

---

**Option B: Eval contamination**

Simulate a scenario where your test data "leaked" into the model's training.

How: create a dataset of 20 questions. Fine-tune a small model (or use few-shot prompting to simulate) on half of them. Evaluate on all 20. Does the model score higher on the "seen" questions? By how much? How would you detect contamination in practice?

Note: you can simulate this without actual fine-tuning by splitting your dataset and including half as examples in the system prompt.

---

**Option C: Human-AI disagreement**

Build a labelling interface, label 20 examples yourself, compare to your LLM judge.

How: write a simple CLI labelling tool (shows question + answer, you type 1–5). Label 20 samples from your Challenge 3 dataset. Compute Cohen's kappa between your scores and the judge's scores. Look at every disagreement — is the judge wrong? Are you? The cases where you disagree most are where the eval is least trustworthy.

---

**Option D: Multi-turn eval**

All your evals so far are single-turn. Real conversations aren't.

How: design an eval for a 3-turn conversation (user asks, model responds, user follows up, model responds, user asks a clarifying question, model responds). How do you score this? Scoring the final turn ignores the coherence of the conversation. Scoring each turn ignores dependencies between turns. Build something, see where it breaks.

---

### Writeup · 6–7pm

Write `docs/FAILURE_MODES.md` up fully. Answer these questions:

1. What does your framework do well?
2. What can it not do at all?
3. What would you design differently if you started over?
4. What's the most surprising thing you learned?
5. If you were building evals at a company, what would you build next?

This document is the most valuable output of the weekend.
