# Failure modes

A living document. Add to this every time something surprises you — a scorer that misfired, a model that behaved unexpectedly, a design decision that turned out to be wrong.

This document is the most valuable output of the weekend.

---

## Template for each entry

```
### [date/time] · [challenge number] · short title

**What happened**: describe what you observed
**Why it happened**: your hypothesis
**What it means**: implication for your framework or for evals generally
**How you'd fix it**: what you'd do differently
```

---

## Saturday entries

<!-- Fill these in as you go -->

### Challenge 1 · harness design

_Add entries here as you build_

### [date/time] · 1 · short title

**What happened**: exact or normalised useless unless you prompt llm to only return very specific format
**Why it happened**: llms are chatty and add text around correct answer
**What it means**: you need to be explicit in the input what output format is needed - what is 1+1 will fail to evaluate to "2" unless you say "ONLY return the number value"
**How you'd fix it**: be explicit in the prompt

### [date/time] · 1 · short title

**What happened**: failure to access llm doesnt cause endless hang
**Why it happened**: evals run, retry up to max retries and then output results
**What it means**: reliable evals completion and not infinite - worst case is max retries OR max retries with last success taking max timeout time.
**How you'd fix it**: retry logic in Runner.run

### [date/time] · 2 · short title

**What happened**: on scorer refusal -> better to return None than 0.0
**Why it happened**: 
**What it means**: this way we have distinction between failure to run (non-llm issue) and failure off llm to get correct answer and can debug the infra and have those issues flagged vs llm quality issues.
**How you'd fix it**:

### [date/time] · 2 · short title

**What happened**: test should be added to pure functions as they are written. anything touched by api or the dataset should be validated by running it and checking manually.
**Why it happened**: 
**What it means**: 
**How you'd fix it**:

### [date/time] · 5 · short title

**What happened**: specifying the output format needed in eval input, for example by adding, ""date" (YYYY-MM-DD string), "amount" (number). Return JSON only, no other text." allows normalised scorer to be used.
**Why it happened**: llm has no chance if it isn't explicitly told the format needed.
**What it means**: can save time and tokens with cascade/tiered scorer with first tier being normalised scorer - still too fragile for just exact match scorer or normalised!
**How you'd fix it**:

### [date/time] · 5 · short title

**What happened**: scorer brittleness - missing or extra 0 - 1240.0 vs 1240.00 can throw off exact or normalised scorers...
**Why it happened**: 
**What it means**: expensive llm as judge needed - fast step misses near miss...
**How you'd fix it**:

### [date/time] · 5 · short title

**What happened**: sometimes llm or judge timeout.
**Why it happened**: 
**What it means**: shows as an error
**How you'd fix it**:

### [date/time] · 5 · short title

**What happened**: sometimes small models fail to format correctly - especially JSON
**Why it happened**: 
**What it means**: 
**How you'd fix it**: 1. be explicit in prompt, 2. sometimes there is a json format mode to set to true, or 3. add function to detect and try and salvage bad json

### [date/time] · 5 · short title

**What happened**: sometimes llms speak to much - "ignore in one sentence" or ignore token limits and just get truncated - which can be fatal for JSON formatting as never closed.
**Why it happened**: 
**What it means**: at worst: error parsing as incomplete json. at best: answer is truncated.
**How you'd fix it**: add detector and close the JSON.

### Challenge 2 · deterministic scoring

_Add entries here as you build_

### Challenge 3 · LLM-as-judge

_Add entries here as you build. Specific things to watch for:_
- [ ] Position bias observed? (Y/N, with example)
- [ ] Verbosity bias observed? (Y/N, with example)
- [ ] Criteria sensitivity observed? (Y/N, with example)

### Challenge 4 · benchmark harness

_Add entries here as you build_

---

## Sunday entries

### Challenge 5 · dataset curation

_Add entries here as you build_

### Challenge 6 · sensitivity analysis

_Add entries here. Include your variance numbers:_
- Exact match scorer variance: 
- Schema scorer variance: 
- LLM judge variance: 

### Challenge 7 · RAG eval suite

_Add entries here as you build_

### Challenge 8 · open problem

_Add entries here as you build_

---

## Known failure modes (pre-populated from the literature)

These are documented problems you should expect to encounter. Check them off when you observe them yourself.

- [ ] **Scorer brittleness** — exact match fails on semantically correct answers due to formatting
- [ ] **LLM judge position bias** — judge prefers the answer presented first
- [ ] **LLM judge verbosity bias** — judge prefers longer answers regardless of quality
- [ ] **LLM judge self-preference** — a model used as judge tends to prefer outputs from the same model family
- [ ] **Eval score variance** — small prompt changes cause large score swings
- [ ] **Dataset contamination** — test examples that appear in training inflate scores
- [ ] **Reward hacking** — model optimises for eval metric without improving on the underlying task
- [ ] **Metric-task misalignment** — high eval scores don't predict good user experience
- [ ] **Distribution gap** — eval dataset doesn't represent real-world input distribution
- [ ] **Partial credit blindness** — binary scorers can't distinguish "almost right" from "completely wrong"
- [ ] **Faithfulness-quality tension** — faithful answers and high-quality answers aren't always the same
- [ ] **Latency-quality tradeoff** — stronger models score higher but slower; eval doesn't capture UX cost
