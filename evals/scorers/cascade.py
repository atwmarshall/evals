from __future__ import annotations

from evals.core import ScorerContext
from evals.scorers.exact import normalised_match
from evals.scorers.llm_judge import LLMJudgeScorer


class CascadeScorer:
    """Two-tier scorer: normalised_match first, LLMJudgeScorer as fallback.

    Returns 1.0 immediately if the completion is a normalised match — no LLM
    call made. Only falls through to the judge when the answer isn't obviously
    correct. Useful for datasets that mix exact answers with open-ended ones.

    Returns float | None — None propagates from the judge on parse failure.
    """

    def __init__(self, judge: LLMJudgeScorer) -> None:
        self._judge = judge

    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float | None:
        if normalised_match(completion, expected, ctx) == 1.0:
            return 1.0
        return self._judge(completion, expected, ctx)
