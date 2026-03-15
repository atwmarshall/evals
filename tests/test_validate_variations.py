from __future__ import annotations

import json
from pathlib import Path
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


class TestSaveVariations:
    def test_writes_jsonl_per_variation(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        validated = {"baseline": ds, "rephrase": ds}
        original = {"baseline": ds, "rephrase": ds}
        out = gen.save_variations(validated, original, "datasets/test.jsonl", threshold=0.8, output_dir=tmp_path)
        assert (tmp_path / "rephrase.jsonl").exists()
        assert not (tmp_path / "baseline.jsonl").exists()

    def test_jsonl_content_is_correct(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        validated = {"rephrase": ds}
        original = {"rephrase": ds}
        gen.save_variations(validated, original, "datasets/test.jsonl", threshold=0.8, output_dir=tmp_path)
        lines = (tmp_path / "rephrase.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["id"] == "a"
        assert obj["input"] == "input a"
        assert obj["expected"] == "expected a"

    def test_generation_metadata_written(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        validated = {"rephrase": Dataset(samples=[ds.samples[0]])}   # one discarded
        original = {"rephrase": ds}
        gen.save_variations(validated, original, "datasets/test.jsonl", threshold=0.8, output_dir=tmp_path)
        meta = json.loads((tmp_path / "generation_metadata.json").read_text())
        assert meta["variation_model"] == "mistral:7b"
        assert meta["threshold"] == 0.8
        assert meta["discard_counts"]["rephrase"] == 1
        assert meta["sample_counts"]["rephrase"] == 1

    def test_default_output_dir_uses_date_and_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a")
        validated = {"rephrase": ds}
        original = {"rephrase": ds}
        out = gen.save_variations(validated, original, "datasets/challenge2/extraction.jsonl", threshold=0.8)
        assert "extraction" in str(out)
        assert "mistral_7b" in str(out)
        assert out.exists()


class TestLoadVariations:
    def test_loads_saved_variations(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        ds = _make_dataset("a", "b")
        validated = {"rephrase": ds, "formal": ds}
        original = {"rephrase": ds, "formal": ds}
        gen.save_variations(validated, original, "datasets/test.jsonl", threshold=0.8, output_dir=tmp_path)
        loaded = VariationGenerator.load_variations(tmp_path)
        assert set(loaded.keys()) == {"rephrase", "formal"}
        assert len(loaded["rephrase"].samples) == 2

    def test_round_trip_preserves_sample_fields(self, tmp_path):
        gen = _gen()
        gen.model = "mistral:7b"
        sample = Sample(id="x", input="hello", expected="world", metadata={"cat": "test"})
        ds = Dataset(samples=[sample])
        gen.save_variations({"rephrase": ds}, {"rephrase": ds}, "datasets/test.jsonl", 0.8, tmp_path)
        loaded = VariationGenerator.load_variations(tmp_path)
        s = loaded["rephrase"].samples[0]
        assert s.id == "x"
        assert s.input == "hello"
        assert s.expected == "world"
        assert s.metadata["cat"] == "test"

    def test_raises_if_directory_missing(self):
        with pytest.raises(FileNotFoundError):
            VariationGenerator.load_variations("/nonexistent/path/xyz")

    def test_raises_if_no_jsonl_files(self, tmp_path):
        with pytest.raises(ValueError, match="no JSONL files"):
            VariationGenerator.load_variations(tmp_path)
