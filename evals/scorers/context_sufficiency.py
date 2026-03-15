from __future__ import annotations

import re

from evals.core import DatasetScorer, ScorerContext

_OVERLAP_THRESHOLD = 0.35


class ContextSufficiencyScorer(DatasetScorer):
    """Deterministic dataset-quality scorer.

    Measures word-level overlap between the expected answer and the combined
    context text. Expected answers are synthesized/paraphrased — they're never
    verbatim substrings of context chunks, so a fraction-of-tokens-matched
    heuristic is used instead.

    Returns 1.0 if overlap >= 0.35, 0.0 if below, None if context is absent.

    Completion is structurally absent from the signature — this scorer measures
    whether the dataset's context is sufficient to support the expected answer,
    not whether the model produced a good answer.

    For rag-006 (Canberra absent from context): overlap=0% → 0.0.
    For all other RAG samples: technical terms from expected appear in context
    → overlap >= 0.35 → 1.0.
    """

    def __call__(self, expected: str, ctx: ScorerContext) -> float | None:
        context = ctx.metadata.get("context")
        if context is None:
            return None

        # Defensive: handle plain string context
        if isinstance(context, str):
            chunks = [context]
        else:
            chunks = list(context)

        if not chunks:
            return None

        context_text = " ".join(str(c) for c in chunks).lower()
        context_tokens = set(re.findall(r"\b\w+\b", context_text))

        expected_tokens = set(re.findall(r"\b\w+\b", expected.lower()))
        if not expected_tokens:
            return 0.0

        overlap = len(expected_tokens & context_tokens) / len(expected_tokens)
        return 1.0 if overlap >= _OVERLAP_THRESHOLD else 0.0
