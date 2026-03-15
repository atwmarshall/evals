from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evals.core import Dataset, Sample
from evals.variation_generator import VariationGenerator


def _make_dataset(*ids: str) -> Dataset:
    return Dataset(samples=[
        Sample(id=i, input=f"input {i}", expected=f"expected {i}", metadata={})
        for i in ids
    ])


def _gen() -> VariationGenerator:
    """VariationGenerator with no Ollama client — validate_variations doesn't need it."""
    return VariationGenerator.__new__(VariationGenerator)


class TestValidateVariations:
    def test_baseline_passes_through_unchanged(self):
        ds = _make_dataset("a", "b")
        result = _gen().validate_variations({"baseline": ds}, validation_scorer=MagicMock(return_value=1.0))
        assert result["baseline"] is ds

    def test_samples_above_threshold_are_kept(self):
        ds = _make_dataset("a", "b")
        scorer = MagicMock(return_value=1.0)
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer, threshold=0.8)
        assert [s.id for s in result["rephrase"].samples] == ["a", "b"]

    def test_samples_below_threshold_are_discarded(self):
        ds = _make_dataset("a", "b")
        scorer = MagicMock(side_effect=[1.0, 0.5])
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer, threshold=0.8)
        assert [s.id for s in result["rephrase"].samples] == ["a"]

    def test_sample_at_threshold_is_kept(self):
        ds = _make_dataset("a")
        scorer = MagicMock(return_value=0.8)
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer, threshold=0.8)
        assert len(result["rephrase"].samples) == 1

    def test_none_score_discards_sample(self):
        ds = _make_dataset("a")
        scorer = MagicMock(return_value=None)
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer)
        assert len(result["rephrase"].samples) == 0

    def test_scorer_exception_discards_sample(self):
        ds = _make_dataset("a", "b")
        scorer = MagicMock(side_effect=[RuntimeError("api down"), 1.0])
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer)
        assert [s.id for s in result["rephrase"].samples] == ["b"]

    def test_empty_variation_key_still_present(self):
        ds = _make_dataset("a", "b")
        scorer = MagicMock(return_value=0.0)
        result = _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer)
        assert "rephrase" in result
        assert len(result["rephrase"].samples) == 0

    def test_scorer_called_with_expected_as_completion(self):
        ds = _make_dataset("a")
        scorer = MagicMock(return_value=1.0)
        _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer)
        call_args = scorer.call_args
        completion, expected, ctx = call_args.args
        assert completion == "expected a"
        assert expected == "expected a"
        assert ctx.input == "input a"

    def test_ctx_includes_sample_id(self):
        ds = _make_dataset("x")
        scorer = MagicMock(return_value=1.0)
        _gen().validate_variations({"rephrase": ds}, validation_scorer=scorer)
        _, _, ctx = scorer.call_args.args
        assert ctx.metadata["id"] == "x"

    def test_multiple_variation_types_filtered_independently(self):
        ds = _make_dataset("a", "b")
        # synonym_swap: both pass; rephrase: only "a" passes
        synonym_scorer = MagicMock(side_effect=[1.0, 1.0])
        rephrase_scorer = MagicMock(side_effect=[1.0, 0.3])

        call_count = [0]

        def scorer(completion, expected, ctx):
            call_count[0] += 1
            # first two calls are synonym_swap, next two are rephrase
            if call_count[0] <= 2:
                return synonym_scorer(completion, expected, ctx)
            return rephrase_scorer(completion, expected, ctx)

        variations = {"synonym_swap": ds, "rephrase": ds}
        result = _gen().validate_variations(variations, validation_scorer=scorer)
        assert len(result["synonym_swap"].samples) == 2
        assert len(result["rephrase"].samples) == 1
