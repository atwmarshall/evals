from __future__ import annotations

from unittest.mock import MagicMock

from evals.core import ScorerContext
from evals.scorers.cascade import CascadeScorer
from evals.scorers.exact import exact_match, normalised_match

_ctx = ScorerContext(input="What is the capital of France?")


def _scorer(score: float | None):
    """Return a fast-scorer stub that always returns `score`."""
    return MagicMock(return_value=score)


class TestCascadeScorer:
    def test_fast_hit_returns_immediately(self):
        mock_judge = MagicMock()
        scorer = CascadeScorer(fast=_scorer(1.0), judge=mock_judge)
        result = scorer("paris", "paris", _ctx)
        assert result == 1.0
        mock_judge.assert_not_called()

    def test_fast_miss_calls_judge(self):
        mock_judge = MagicMock(return_value=0.6)
        scorer = CascadeScorer(fast=_scorer(0.0), judge=mock_judge)
        result = scorer("Lyon", "paris", _ctx)
        assert result == 0.6
        mock_judge.assert_called_once_with("Lyon", "paris", _ctx)

    def test_judge_none_propagates(self):
        mock_judge = MagicMock(return_value=None)
        scorer = CascadeScorer(fast=_scorer(0.0), judge=mock_judge)
        result = scorer("something", "rubric", _ctx)
        assert result is None

    def test_fast_none_falls_through_to_judge(self):
        # If fast scorer returns None, treat as a miss and call judge
        mock_judge = MagicMock(return_value=0.5)
        scorer = CascadeScorer(fast=_scorer(None), judge=mock_judge)
        result = scorer("something", "rubric", _ctx)
        assert result == 0.5
        mock_judge.assert_called_once()

    def test_custom_threshold(self):
        # threshold=0.8: fast score of 0.9 short-circuits
        mock_judge = MagicMock()
        scorer = CascadeScorer(fast=_scorer(0.9), judge=mock_judge, threshold=0.8)
        result = scorer("good enough", "expected", _ctx)
        assert result == 0.9
        mock_judge.assert_not_called()

    def test_custom_threshold_miss(self):
        # threshold=0.8: fast score of 0.5 falls through
        mock_judge = MagicMock(return_value=0.75)
        scorer = CascadeScorer(fast=_scorer(0.5), judge=mock_judge, threshold=0.8)
        result = scorer("partial", "expected", _ctx)
        assert result == 0.75
        mock_judge.assert_called_once()

    def test_exact_match_as_fast_tier(self):
        # Callers choose the fast tier — exact_match is valid
        mock_judge = MagicMock(return_value=0.4)
        scorer = CascadeScorer(fast=exact_match, judge=mock_judge)
        assert scorer("Paris", "Paris", _ctx) == 1.0
        mock_judge.assert_not_called()
        assert scorer("paris", "Paris", _ctx) == 0.4  # case mismatch → judge
        mock_judge.assert_called_once()

    def test_normalised_match_as_fast_tier(self):
        mock_judge = MagicMock()
        scorer = CascadeScorer(fast=normalised_match, judge=mock_judge)
        assert scorer("Paris!", "paris", _ctx) == 1.0
        mock_judge.assert_not_called()


class TestCascadeMetadataOut:
    def test_fast_tier_sets_tier_used_and_fast_score(self):
        ctx = ScorerContext()
        mock_judge = MagicMock()
        scorer = CascadeScorer(fast=_scorer(1.0), judge=mock_judge)
        scorer("ans", "ans", ctx)
        assert ctx.metadata_out["tier_used"] == "fast"
        assert ctx.metadata_out["fast_score"] == 1.0

    def test_judge_tier_sets_tier_used_and_fast_score(self):
        ctx = ScorerContext()
        mock_judge = MagicMock(return_value=0.6)
        scorer = CascadeScorer(fast=_scorer(0.0), judge=mock_judge)
        scorer("ans", "expected", ctx)
        assert ctx.metadata_out["tier_used"] == "judge"
        assert ctx.metadata_out["fast_score"] == 0.0

    def test_fast_score_always_present_in_fast_path(self):
        ctx = ScorerContext()
        scorer = CascadeScorer(fast=_scorer(1.0), judge=MagicMock())
        scorer("ans", "expected", ctx)
        assert "fast_score" in ctx.metadata_out

    def test_fast_score_always_present_in_judge_path(self):
        ctx = ScorerContext()
        mock_judge = MagicMock(return_value=0.8)
        scorer = CascadeScorer(fast=_scorer(0.5), judge=mock_judge, threshold=0.8)
        scorer("ans", "expected", ctx)
        assert "fast_score" in ctx.metadata_out
