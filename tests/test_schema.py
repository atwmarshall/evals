from __future__ import annotations

import pytest

from evals.scorers.schema import JSONSchemaScorer

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


class TestJSONSchemaScorer:
    def test_valid_json_passes_schema(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('{"name": "Alice", "age": 30}', "") == 1.0

    def test_valid_json_missing_required_field(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('{"age": 30}', "") == 0.5

    def test_valid_json_wrong_type(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('{"name": 123}', "") == 0.5

    def test_not_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score("not json at all", "") == 0.0

    def test_empty_string(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score("", "") == 0.0

    def test_partial_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('{"name": "Al', "") == 0.0

    def test_json_with_leading_whitespace(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('  {"name": "Bob"}  ', "") == 1.0

    def test_empty_object_against_schema_with_required(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score("{}", "") == 0.5

    def test_array_schema(self):
        scorer = JSONSchemaScorer({"type": "array", "items": {"type": "integer"}})
        assert scorer.score("[1, 2, 3]", "") == 1.0
        assert scorer.score('["a", "b"]', "") == 0.5
        assert scorer.score("not an array", "") == 0.0

    def test_expected_arg_ignored(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer.score('{"name": "X"}', "completely ignored") == 1.0
