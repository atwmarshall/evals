# Open problems in LLM evaluation

These are genuinely unsolved problems — not things with clean answers. Pick one and go deep.

---

## 1. Reward hacking

**The problem**: when you optimise a model against an eval metric, the model learns to maximise the metric rather than the underlying quality it's supposed to measure. The eval score goes up; real quality doesn't. Sometimes quality goes down.

**Why it's hard**: you can't define a metric that perfectly captures quality. Every proxy metric has gaps, and a sufficiently capable model will find and exploit those gaps.

**Real examples**:
- RLHF models trained against human preference ratings learned to produce long, confident-sounding answers — because raters preferred them — even when shorter, more uncertain answers were more correct.
- Code generation models trained against test pass rates learned to generate code that special-cases the tests.
- Summarisation models trained against ROUGE scores learned to copy sentences verbatim rather than paraphrase.

**What to explore**: build an eval, then write a system prompt that tells the model what the eval measures and instructs it to maximise the score. Run both the original model and the "gaming" model. How large is the gap? Can you design an eval that's harder to game?

**Key reading**: "Goodhart's Law" (when a measure becomes a target, it ceases to be a good measure). Amodei et al., "Concrete Problems in AI Safety" (2016), section on reward hacking.

---

## 2. Eval contamination

**The problem**: if your test examples appeared in the model's training data, the model has "memorised" the answers and your eval scores are inflated. You're measuring memorisation, not generalisation.

**Why it's hard**: you don't have access to training data for most frontier models. You can't know if your eval is contaminated. Even if a specific example wasn't in training, semantically similar examples might have been.

**Real examples**:
- GPT-4's reported performance on many standard benchmarks (MMLU, HumanEval) was later questioned when it emerged that these benchmarks had been widely discussed online before the training cutoff.
- Several "new" benchmarks achieved state-of-the-art within months of release, faster than genuine capability improvements would explain.

**What to explore**: create two dataset splits. Include one split verbatim in a system prompt as "examples" (simulating contamination). Evaluate on both splits. What's the score gap? How would you detect this in practice without access to training data?

**Detection heuristics** (imperfect but useful):
- Unusually high scores on a new dataset with no fine-tuning
- Model produces answer before "reasoning" through the problem
- Scores don't degrade gracefully with increasing difficulty

---

## 3. Human-AI disagreement

**The problem**: LLM judges and human raters disagree — sometimes a lot. The question is: who's right? And what does "right" even mean for subjective quality?

**Why it's hard**: human raters also disagree with each other. Inter-rater agreement (measured by Cohen's kappa) is typically 0.4–0.6 for subjective quality tasks — "moderate" agreement. LLM judges often hit similar numbers. But the *patterns* of disagreement differ.

**Real examples**:
- LLM judges tend to prefer longer, more structured answers. Humans often prefer concise ones.
- LLM judges are more consistent (same answer gets same score each time). Humans are more sensitive to context and framing.
- Humans catch factual errors that LLM judges miss (especially hallucinations about obscure topics).
- LLM judges catch stylistic issues humans overlook when they agree with the content.

**What to explore**: label 20 examples yourself. Compute kappa against your LLM judge. Examine every disagreement. Categorise them: is the judge systematically wrong? Are you? Build a "meta-eval" — criteria for when to trust the judge vs. when to override it.

**Key question**: for your specific task, which failure mode is more costly — false positives (judge approves bad answers) or false negatives (judge rejects good answers)?

---

## 4. Multi-turn evaluation

**The problem**: almost all eval benchmarks are single-turn (one question, one answer). Real LLM deployments are multi-turn conversations. Single-turn evals don't measure the things that matter most: coherence across turns, memory, ability to recover from errors, ability to handle clarification requests.

**Why it's hard**: the evaluation space is exponential. A 5-turn conversation has ~5 decision points, each with many possible paths. You can't enumerate all trajectories. And scoring a full conversation is much harder than scoring a single answer — what matters, the final turn? The average? The worst turn?

**Real examples**:
- A model that scores well on single-turn QA may fail badly when the user follows up with "are you sure?" or "can you simplify that?"
- Summarisation models that produce great single-turn summaries may lose coherence when asked to refine iteratively.
- Code assistants that write correct code on the first try may produce garbage when the user asks for a modification.

**What to explore**: design a 3-turn eval. Three turns: initial question → follow-up → clarification. Write a scorer that evaluates the full trajectory. Hard questions to answer:
- How do you score turn 2 if turn 1 was wrong?
- How do you handle branching trajectories?
- What's your "expected" for an open-ended multi-turn conversation?

---

## 5. Eval validity (the meta-problem)

**The problem**: how do you know your eval is measuring what you think it's measuring? This is called "construct validity" in psychometrics. It's the hardest problem in evals.

**Why it's hard**: you need ground truth to validate your eval, but if you had ground truth you wouldn't need an eval.

**Real examples**:
- An eval for "reasoning ability" might actually be measuring "ability to write reasoning-shaped text."
- An eval for "helpfulness" might actually be measuring "compliance with instructions."
- An eval for "safety" might actually be measuring "willingness to refuse."

**Partial solutions**:
- **Predictive validity**: does your eval score predict real-world outcomes? (Hard to measure)
- **Convergent validity**: does your eval correlate with other evals of the same construct? (Circular if those evals have the same problems)
- **Face validity**: do experts agree the eval measures what you say it measures? (Subjective but useful)
- **Sensitivity analysis**: does your eval detect known improvements? (Build a deliberately better and worse model, check your eval distinguishes them)

**What to explore**: take your best scorer from Saturday. Make a list of 5 things it might be measuring that aren't what you intended. Design one test for each to check. This is harder than building the scorer — and more important.
