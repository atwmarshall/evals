# Failure modes

A living document. Add to this every time something surprises you — a scorer that misfired, a model that behaved unexpectedly, a design decision that turned out to be wrong.


---

## Template for each entry

```
### [date/time] · short title

**What happened**: describe what you observed
**Why it happened**: your hypothesis
**What it means**: implication for your framework or for evals generally
**How you'd fix it**: what you'd do differently
```

---

## Saturday entries

<!-- Fill these in as you go -->

### Harness design

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
**Why it happened**: model hits max_tokens mid-output; JSON is a streaming format with no graceful truncation point. Small models are more susceptible.
**What it means**: a structurally-sound answer gets scored 0.0, indistinguishable from a wrong answer. For judges, it shows as parse_failure in the run results.
**How you'd fix it**: add detector and close the JSON.
**Status**: implemented — `evals/scorers/_json_utils.py::_repair_truncated_json()` is applied automatically by `JSONSchemaScorer` and `LLMJudgeScorer`.

### [date/time] · 6 · short title
**What happened**: to measure scorer reliability, we generate datasets with variation but we shouldnt use the judge model for this as that's the model that is our scorer!
**Why it happened**: shouldnt be the model we are evaluating either
**What it means**: Add a VARIATION_MODEL env var to .env.example and make it fail loudly if judge model = variation model.
**How you'd fix it**:

### Deterministic scoring

### 2026-03-15 · 2/5 · partial credit blindness in run output

**What happened**: `JSONSchemaScorer` returns 0.75 for "one field failed validation" and `LLMJudgeScorer` returns 0.75 for "good but not perfect". Both showed as `type=fail` in the results table, collapsing near-misses and genuine failures into one category — different problems that need different responses.
**Why it happened**: The `type` column was a binary pass/fail derived from `score == 1.0`, with no intermediate tier. The column name `type` also gave no indication of what it represented.
**What it means**: You can't act on results you can't distinguish. A `partial` outcome (score 0.5–1.0) needs investigation of why the model is close but not exact; a `fail` outcome (score < 0.5) needs a different response entirely (prompt redesign, model swap). Mixing them hides the signal.
**How you'd fix it**: Add a `partial` outcome tier. Rename `type` → `outcome`. Column order: `id score outcome latency_ms`. Add `--strict` flag to `show.py` to filter to score < 0.5 only.
**Status**: implemented — `_outcome()` in `show.py`, `_outcome_str()` in `reporters.py`, `--strict` / `-s` flag added.

### 2026-03-15 · General embeddings too semantic for context sufficiency

**What happened**: rag-006 scored 0.66 with embedding cosine similarity.
Canberra not in context but semantically related to Sydney/Melbourne.
**Why it happened**: nomic-embed-text captures topical similarity not
answer presence. All Australian cities cluster together in embedding space.
**What it means**: embedding similarity measures topical relatedness,
not logical entailment. Wrong tool for "is this specific fact in this context?"
**How you'd fix it**: LLM-based YES/NO entailment check, or RAGAS-style
atomic statement decomposition.

### 2026-03-15 · Context sufficiency false negative — paraphrase mismatch

**What happened**: `rag-007` scored 0.0. Context chunk [0] contains all three RAG failure modes. Expected answer names the same three modes but uses "answer quality failure" where the context says "synthesis failure." LLM judge said NO.
**Why it happened**: the YES/NO prompt asks "does the context contain enough information to answer this question" — the LLM interpreted this as lexical matching, not semantic entailment. Same concept, different words, wrong verdict.
**What it means**: LLM-based context sufficiency checking is sensitive to paraphrase direction. The scorer is doing lexical matching in disguise. This is the mirror image of the embedding failure — embeddings were too loose (related facts scored high), the LLM check is too strict (paraphrased facts score low).
**How you'd fix it**: add "even if the answer would use different words than the context" to the sufficiency prompt. Or fix the expected answer to match context wording exactly ("synthesis failure" not "answer quality failure"). The deeper fix: align expected answers with context wording at dataset creation time — don't paraphrase if the context uses specific terminology.

---

### 2026-03-15 · Dataset bug — expected answer requires arithmetic not in context

**What happened**: `rag-004` scored 0.0. Context states "0.75 words per token." Expected answer states "1.3 tokens per word." The LLM correctly said NO — 1.3 tokens per word requires computing 1/0.75 = 1.33, which no context chunk states explicitly.
**Why it happened**: the expected answer was written by deriving a fact from the context rather than quoting or closely paraphrasing it. The scorer correctly identified the context as insufficient for the stated expected answer.
**What it means**: this is a dataset quality bug, not a scorer failure. The scorer worked correctly — it exposed a flaw in the ground truth label. High-quality eval datasets must have expected answers that are directly derivable from context without implicit calculation or inference steps. If derivation is required, the context must include the derived fact explicitly.
**How you'd fix it**: either add a context chunk stating "approximately 1.3 tokens per word" explicitly, or change the expected answer to match what the context directly supports: "approximately 0.75 words per token, or about 4 characters per token." Lesson: run context sufficiency on your dataset before running any model evals — it catches label errors cheaply.

*MAJOR:*
### 2026-03-15 · Chain-of-thought improves scorer accuracy

**What happened**: adding "provide reasoning if NO" to the context sufficiency
prompt changed mean_score from 0.700 to 0.900. rag-003 and rag-007 flipped
from false negatives to correct passes.
**Why it happened**: requiring the model to articulate the gap before committing
to NO forces it to check the context carefully. This is chain-of-thought prompting
— the reasoning text becomes additional context that influences the final answer.
**What it means**: prompt design affects scorer accuracy, not just model accuracy.
A scorer that asks for reasoning is a more reliable scorer. This is why
LLMJudgeScorer asks for reasoning — not just for debugging but for accuracy.
**The implication**: always ask for reasoning before the answer in judge/scorer
prompts. "Answer first, then explain" is worse than "explain then answer" for
accuracy. The reasoning is doing real work, not just providing a trace.
**The practical takeaway for your framework:**
Your LLMJudgeScorer already does this correctly — it asks for reasoning before the score. Your context sufficiency scorer now does too. Any new judge-style scorer you add should follow the same pattern: reasoning before answer, not after.
The exact prompt structure that works:
[task description]
[ask for reasoning if negative/low]
[then ask for the answer]
Not:
[task description]  
[ask for the answer]
[optionally explain]
Order matters because of left-to-right generation. The reasoning has to come before the answer token to influence it.
**THE TRACE - see RAG-003 and RAG-007!**
$uv run scripts/run_eval.py --dataset datasets/rag/rag_qa.jsonl --scorer context-sufficiency
Running eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:03<00:00,  2.87it/s]
id         score  outcome      latency_ms  error
-------  -------  ---------  ------------  -------
rag-001        1  pass                  0
rag-002        1  pass                  0
rag-003        0  fail                  0
rag-004        1  pass                  0
rag-005        1  pass                  0
rag-006        0  fail                  0
rag-007        0  fail                  0
rag-008        1  pass                  0
rag-009        1  pass                  0
rag-010        1  pass                  0

mean_score=0.700  p50_latency=0ms  p95_latency=0ms (n=10 ⚠)  api_errors=0  parse_failures=0  error_rate=0.0%

Saved → results/runs/2026-03-15/235412_llama3.2_3b_rag_qa_context-sufficiency
$uv run scripts/show.py results/runs/2026-03-15/235412_llama3.2_3b_rag_qa_context-sufficiency --id rag-006
RUN  model=llama3.2:3b  dataset=rag_qa  scorer=context-sufficiency  2026-03-15T23:54:12
     mean_score=0.700  p50=0ms  errors=0/10

id=rag-006  score=0.0  latency=0ms  outcome=fail
expected:  Canberra
error: None

context:
  [0] Australia is a country in the southern hemisphere.
  [1] Sydney is the largest city in Australia and a major financial centre.
  [2] Melbourne is known for its cultural scene and is the second-largest city.

reasoning:  NO

completion: (not applicable — dataset scorer)
$uv run scripts/run_eval.py --dataset datasets/rag/rag_qa.jsonl --scorer context-sufficiency
Running eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:11<00:00,  1.12s/it]
id         score  outcome      latency_ms  error
-------  -------  ---------  ------------  -------
rag-001        1  pass                  0
rag-002        1  pass                  0
rag-003        1  pass                  0
rag-004        1  pass                  0
rag-005        1  pass                  0
rag-006        0  fail                  0
rag-007        1  pass                  0
rag-008        1  pass                  0
rag-009        1  pass                  0
rag-010        1  pass                  0

mean_score=0.900  p50_latency=0ms  p95_latency=0ms (n=10 ⚠)  api_errors=0  parse_failures=0  error_rate=0.0%

Saved → results/runs/2026-03-15/235951_llama3.2_3b_rag_qa_context-sufficiency

$uv run scripts/show.py results/runs/2026-03-15/235951_llama3.2_3b_rag_qa_context-sufficiency --id rag-006
RUN  model=llama3.2:3b  dataset=rag_qa  scorer=context-sufficiency  2026-03-15T23:59:51
     mean_score=0.900  p50=0ms  errors=0/10

id=rag-006  score=0.0  latency=0ms  outcome=fail
expected:  Canberra
error: None

context:
  [0] Australia is a country in the southern hemisphere.
  [1] Sydney is the largest city in Australia and a major financial centre.
  [2] Melbourne is known for its cultural scene and is the second-largest city.

reasoning:  The context does not provide information about the capital city of Australia.

completion: (not applicable — dataset scorer)
**END**


### LLM-as-judge

_Add entries here as you build. Specific things to watch for:_
- [ ] Position bias observed? (Y/N, with example)
- [ ] Verbosity bias observed? (Y/N, with example)
- [ ] Criteria sensitivity observed? (Y/N, with example)

### Benchmark harness

_Add entries here as you build_

---

## Sunday entries

### Dataset curation

_Add entries here as you build_

### Sensitivity analysis

_Add entries here. Include your variance numbers:_
- Exact match scorer variance: 
- Schema scorer variance: 
- LLM judge variance: 

### RAG eval suite

### 2026-03-15 · 7 · ROUGE-1 threshold tuned on the test set

**What happened**: `ContextSufficiencyScorer` used word-level token overlap (ROUGE-1) with `threshold=0.35`. The threshold was chosen because it cleanly separated rag-006 (overlap=0%) from rag-010 (overlap=38%) on 10 samples.
**Why it happened**: Binary classification needs a threshold. The obvious move is to pick the one that separates passing and failing samples — which is exactly what we did against the only data we had.
**What it means**: The threshold was calibrated on the test set. Any threshold tuned on eval data is no longer an independent test — adding one new sample could require retuning. More fundamentally, ROUGE-1 is lexical: "temperature 0 makes outputs deterministic" and "use temperature 0 for determinism" share few tokens but mean the same thing. Token overlap penalises correct paraphrase.
**How you'd fix it**: Replace with embedding cosine similarity. No threshold to tune — the score is continuous and the gap between semantically present and semantically absent answers is natural, not engineered.
**Status**: implemented (embeddings) — then superseded. See next entry.

### 2026-03-15 · 7 · Embedding cosine similarity too semantic for entailment checking

**What happened**: rag-006 scored ~0.59 with `nomic-embed-text` cosine similarity. "Canberra" vs chunks about Sydney and Melbourne — semantically related (all Australian cities) but Canberra is not derivable from the context.
**Why it happened**: Embeddings measure topical relatedness, not logical entailment. Australian city names cluster together in embedding space regardless of whether any specific city appears in the text.
**What it means**: Cosine similarity answers "are these about the same topic?" not "does this text contain enough information to produce this answer?" Wrong question for context sufficiency checking.
**How you'd fix it**: LLM YES/NO entailment check. An LLM understands that "Canberra" is not derivable from context about Sydney and Melbourne even when embeddings disagree.
**Status**: implemented — `ContextSufficiencyScorer` now uses a chat call asking `Does the following context contain enough information to answer this question? Answer only YES or NO.` Returns 1.0/0.0 on YES/NO, None on unexpected response (e.g. "Not enough information"). numpy dependency removed.

### Open problem

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
- [x] **Partial credit blindness** — binary scorers can't distinguish "almost right" from "completely wrong"
- [ ] **Faithfulness-quality tension** — faithful answers and high-quality answers aren't always the same
- [ ] **Latency-quality tradeoff** — stronger models score higher but slower; eval doesn't capture UX cost
