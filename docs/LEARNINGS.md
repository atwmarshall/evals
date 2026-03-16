# Learnings

Key takeaways from building an eval framework from scratch. These are things
that can't be read off a diagram — they were discovered by hitting walls.

---

## Scorer design

### 1. The scorer contract had to evolve — and the reason why matters

The original contract `(completion: str, expected: str) -> float` was clean but
broke the moment the judge needed to see the original question. Five options were
considered. The right answer was `ScorerContext` — a typed dataclass the runner
populates and scorers optionally inspect:

```python
(completion: str, expected: str, ctx: ScorerContext) -> float
```

Pure scorers ignore `ctx`. The judge uses `ctx.input`. The cascade uses
`ctx.metadata`. The contract is load-bearing for testability — scorers remain
pure functions, composable and testable in isolation.

**The principle**: don't break the core contract to patch one scorer's context
needs. Extend it with an opt-in side channel.

Industry parallel: `inspect_ai`'s `TaskState`, Braintrust's `Eval` object — both
are typed context objects the runner populates and scorers optionally inspect.
You hit the same design pressure and arrived at the same solution independently.

---

### 2. Two booleans that can't vary independently should be one field

Early design had `format_passed: bool` and `repaired: bool` in `scorer_metadata`.
The impossible state (`format_passed=True, repaired=True`) was a design smell.
Three real states → one enum:

```python
"format_status": "clean" | "repaired" | "repair_failed"
```

**The principle**: when you find yourself writing two booleans, ask whether all
four combinations are actually possible. If not, you have one field with N states,
not two independent booleans. This applies everywhere — not just evals.

Derived rates computed at aggregation time, not in the scorer:
- `clean_rate` — raw format compliance, no intervention
- `format_pass_rate` — production reliability (clean + repaired)
- `repair_failure_rate` — unrecoverable failures, maps directly to score=0.0

---

### 3. DatasetScorer — a different contract for a different purpose

`ContextSufficiencyScorer` was initially wired through `Runner`, which called the
model unconditionally before scoring. The scorer ignored the completion entirely.
This was wrong in two ways: wasted model calls, and misleading `show.py` output
that displayed a completion as if it affected the score.

The fix: a `DatasetScorer` marker class with a two-argument contract:

```python
(expected: str, ctx: ScorerContext) -> float
# completion dropped — structurally absent, not ignored
```

Runner detects `isinstance(scorer, DatasetScorer)` and skips the model call.

**The principle**: don't represent impossible states. If `completion` is never
used, removing it from the signature is more honest than ignoring it. The
two-argument contract makes it *impossible* to accidentally use completion.

**The broader category**: context sufficiency is a *dataset quality metric*, not
a *model quality metric*. Run it before model evals to validate dataset
construction. A bad sample produces misleading model scores regardless of how
good your scorer is. If context sufficiency is 0.0 for a sample, remove it from
the eval — don't run models against it.

---

### 4. Chain-of-thought in scorer prompts improves accuracy, not just trace quality

Original context sufficiency prompt: binary YES/NO. Mean score: 0.700. Two false
negatives (rag-003, rag-007) — paraphrase mismatches the model scored as NO.

After adding "provide reasoning if NO": mean score jumped to 0.900. Both false
negatives flipped to correct passes. The model said YES with no reasoning shown
— it didn't need to articulate a gap because there wasn't one.

**Why it works**: language models generate left-to-right. When the model writes
a reasoning sentence *before* writing YES/NO, that reasoning is now in its context
window and influences the answer token. Requiring justification for the negative
case forces a reasoning step that improves accuracy on positive cases too — the
model asks itself "would I be able to justify a NO here?" before answering.

**The principle**: in judge/scorer prompts, reasoning before the answer is
chain-of-thought and improves accuracy. Reasoning after the answer is just a
trace. Order matters.

```
# correct — reasoning influences the answer
{"reasoning": "...", "answer": "YES"}

# weaker — reasoning explains a decision already made  
{"answer": "YES", "reasoning": "..."}
```

This is why `LLMJudgeScorer` asks for reasoning before the score. Apply this
pattern to every new judge-style scorer.

---

## Eval reliability

### 5. Scorer brittleness and model brittleness look identical — until sensitivity analysis

Running sensitivity analysis with `--scorer exact` on extraction samples produced
two `unstable` verdicts. The instability wasn't the model degrading under varied
inputs — it was `exact_match` failing on varied phrasings of the same question
because the model produced slightly different JSON formatting.

`c2-000` scored 1.0 on baseline and synonym_swap, 0.0 on concise and formal.
Same model, same content, different output formatting triggered by different input
phrasing. The scorer was the problem, not the model.

**The principle**: sensitivity analysis has two targets — scorer reliability and
model robustness — and they require different experimental designs:

| What varies | What's measured | Name |
|---|---|---|
| Scorer prompt | Scorer consistency | Sensitivity analysis / evaluator reliability |
| Model input | Model consistency | Robustness testing / perturbation testing |

Conflating them produces uninterpretable results. Establish scorer reliability
first. Only run robustness testing with a scorer you've already validated.

---

### 6. The ruler must be validated before the measurement

You can't trust robustness results if your scorer has high variance. If scorer
variance is ±0.1 and model degradation under perturbation is 0.08, the degradation
is within the noise floor — you can't claim the model is fragile.

**The correct order**:
1. Sensitivity analysis → establish scorer noise floor
2. Only if noise floor is acceptable → robustness testing
3. Report robustness results with the scorer noise floor as a confidence bound

This sequencing is standard in measurement science and in eval frameworks — you
validate the instrument before you use it. You built them in this order without
knowing that's what you were doing.

---

## RAG evaluation

### 7. Three implementations of one scorer — each exposed a different failure mode

`ContextSufficiencyScorer` went through three complete rewrites in one session:

**ROUGE-1 token overlap** → threshold overfit to 10 samples, lexical not semantic,
penalises valid paraphrase. Lesson: a threshold tuned on your test set is no longer
an independent test.

**Embedding cosine similarity** → too semantic, related-but-wrong facts inflate
scores. "Canberra" vs "Sydney/Melbourne" scored 0.59 because all are Australian
cities. Lesson: general-purpose embeddings measure topical relatedness, not logical
entailment. Wrong tool for "is this specific fact in this context?"

**LLM YES/NO** (without reasoning) → lexical matching in disguise. "Synthesis
failure" vs "answer quality failure" scored NO despite being the same concept.
Lesson: without chain-of-thought, LLM classifiers revert to surface matching.

**LLM YES/NO with chain-of-thought** → correct results. The reasoning requirement
forced semantic evaluation.

The progression is the learning. Each failure mode was invisible until the fix
for the previous one was in place. This is why iterating on a working system with
real data is more valuable than designing the perfect system upfront.

---

### 8. The faithfulness-quality tension is real and observable

`rag-006` is the canonical example: the context contains no mention of Canberra.
A *faithful* model should say "I can't determine the answer from the context."
A *correct* model says "Canberra."

These are in direct conflict. The eval has to decide which it values more — and
that decision encodes product values, not just technical preference:

- Evaluating a RAG system for a regulated industry (legal, medical) → faithfulness
  wins. Hallucinated true facts are worse than admitting uncertainty.
- Evaluating a general-purpose assistant → quality may win. A confident correct
  answer is better UX than "I don't know."

**The principle**: there is no objectively correct weighting. Writing an eval
forces you to be explicit about values you might otherwise leave implicit. The
discomfort of that decision is the point — it's where real AI engineering happens.

---

### 9. Dataset quality dominates everything — run dataset-level checks first

`rag-004` scored 0.0 on context sufficiency correctly. The expected answer
(`"1.3 tokens per word"`) requires computing 1/0.75 = 1.33 — arithmetic not
explicitly in any context chunk. The scorer was right. The label was wrong.

`rag-007`'s expected answer used "answer quality failure" where the context said
"synthesis failure." Same concept, different words. Two false negatives before the
CoT fix, one correct pass after.

Both are dataset bugs, not model bugs. High-quality eval datasets require:
- Expected answers directly derivable from context without implicit calculation
- Consistent terminology between context and expected answer
- At least one adversarial sample where context genuinely can't support the answer

**The principle**: run context sufficiency and label format validation on your
dataset *before* running any model evals. Garbage in, garbage out — but the
garbage hides behind plausible-looking scores.

---

## Architecture

### 10. Errors are data, not exceptions

The most important architectural decision in the framework: `RunResult.error`
holds the exception string. API errors, scorer errors, parse failures — all stored
as data, never swallowed, never re-raised (except `KeyboardInterrupt`).

This enables:
- `error_rate` in `_summarise()` — what fraction of runs had infrastructure failures
- `parse_failures` separate from `api_errors` — different failure modes, different fixes
- `None` score vs `0.0` score — a judge parse failure and a genuinely bad answer
  look different in the results
- `format_status: "repair_failed"` — unrecoverable format failure vs wrong content

**The `None` vs `0.0` distinction is load-bearing**:
- `score=0.0` — the model was scored and got it wrong
- `score=None` — the scorer couldn't determine a score (infrastructure failure)

Collapsing these to 0.0 would make a model that always times out look identical
to a model that always gives wrong answers. They require completely different fixes.

**The principle**: in evaluation systems, the difference between "couldn't measure"
and "measured and it's bad" is the most important distinction you can preserve.
Design your data model to make it impossible to confuse them.