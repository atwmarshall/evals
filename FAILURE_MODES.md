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
- [ ] **LLM judge self-preference** — Claude judge prefers Claude outputs over other models
- [ ] **Eval score variance** — small prompt changes cause large score swings
- [ ] **Dataset contamination** — test examples that appear in training inflate scores
- [ ] **Reward hacking** — model optimises for eval metric without improving on the underlying task
- [ ] **Metric-task misalignment** — high eval scores don't predict good user experience
- [ ] **Distribution gap** — eval dataset doesn't represent real-world input distribution
- [ ] **Partial credit blindness** — binary scorers can't distinguish "almost right" from "completely wrong"
- [ ] **Faithfulness-quality tension** — faithful answers and high-quality answers aren't always the same
- [ ] **Latency-quality tradeoff** — stronger models score higher but slower; eval doesn't capture UX cost
