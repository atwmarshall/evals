from __future__ import annotations

from collections.abc import Callable

from evals.core import ScorerContext
from evals.scorers.llm_judge import LLMJudgeScorer

ScorerCallable = Callable[[str, str, ScorerContext], float | None]

# Score at or above this threshold short-circuits to the fast tier result.
# 1.0 means only a perfect fast-tier score avoids the judge call.
_DEFAULT_THRESHOLD = 1.0


class CascadeScorer:
    """Two-tier scorer: a fast scorer first, LLMJudgeScorer as fallback.

    If the fast scorer returns a score >= threshold, that score is returned
    immediately — no judge call is made. If it falls below the threshold, the
    judge scores the completion and its result is returned.

    The fast tier is a sufficient-signal check: when normalised string match
    fires, the answer is unambiguously correct and the judge adds nothing. When
    it misses, the judge catches near-misses and partial credit that string
    comparison can't see.

    Returns float | None — None propagates from the judge on parse failure.
    """

    def __init__(
        self,
        fast: ScorerCallable,
        judge: LLMJudgeScorer,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._fast = fast
        self._judge = judge
        self._threshold = threshold

    def set_evaluated_model(self, model_id: str) -> None:
        self._judge.set_evaluated_model(model_id)

    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float | None:
        fast_score = self._fast(completion, expected, ctx)
        if fast_score is not None and fast_score >= self._threshold:
            return fast_score
        return self._judge(completion, expected, ctx)
