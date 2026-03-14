from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evals.core import ScorerContext
from evals.scorers.cascade import CascadeScorer


class TestCascadeScorer:
    def test_exact_match_returns_1_without_calling_judge(self):
        mock_judge = MagicMock()
        scorer = CascadeScorer(judge=mock_judge)
        ctx = ScorerContext(input="What is 2+2?")
        result = scorer("paris", "paris", ctx)
        assert result == 1.0
        mock_judge.assert_not_called()

    def test_normalised_match_short_circuits(self):
        # Normalised match ignores case and punctuation
        mock_judge = MagicMock()
        scorer = CascadeScorer(judge=mock_judge)
        ctx = ScorerContext(input="Name the capital of France.")
        result = scorer("Paris!", "paris", ctx)
        assert result == 1.0
        mock_judge.assert_not_called()

    def test_mismatch_calls_judge(self):
        mock_judge = MagicMock(return_value=0.6)
        scorer = CascadeScorer(judge=mock_judge)
        ctx = ScorerContext(input="Explain gradient descent.")
        result = scorer("It minimises the loss.", "Should explain iterative optimisation.", ctx)
        assert result == 0.6
        mock_judge.assert_called_once_with(
            "It minimises the loss.", "Should explain iterative optimisation.", ctx
        )

    def test_judge_none_propagates(self):
        mock_judge = MagicMock(return_value=None)
        scorer = CascadeScorer(judge=mock_judge)
        ctx = ScorerContext(input="Explain something.")
        result = scorer("some answer", "some rubric", ctx)
        assert result is None

    def test_judge_low_score_propagates(self):
        mock_judge = MagicMock(return_value=0.25)
        scorer = CascadeScorer(judge=mock_judge)
        ctx = ScorerContext(input="What is entropy?")
        result = scorer("I dunno", "Should cover information theory.", ctx)
        assert result == 0.25
