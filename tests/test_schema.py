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
        assert scorer('{"name": "Alice", "age": 30}', "") == 1.0

    def test_valid_json_missing_required_field(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"age": 30}', "") == 0.5

    def test_valid_json_wrong_type(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": 123}', "") == 0.5

    def test_not_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("not json at all", "") == 0.0

    def test_empty_string(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("", "") == 0.0

    def test_partial_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": "Al', "") == 0.0

    def test_json_with_leading_whitespace(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('  {"name": "Bob"}  ', "") == 1.0

    def test_empty_object_against_schema_with_required(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer("{}", "") == 0.5

    def test_array_schema(self):
        scorer = JSONSchemaScorer({"type": "array", "items": {"type": "integer"}})
        assert scorer("[1, 2, 3]", "") == 1.0
        assert scorer('["a", "b"]', "") == 0.5
        assert scorer("not an array", "") == 0.0

    def test_expected_arg_ignored(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('{"name": "X"}', "completely ignored") == 1.0

    def test_markdown_json_fence(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```json\n{"name": "Alice"}\n```', "") == 1.0

    def test_markdown_fence_no_language(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```\n{"name": "Alice"}\n```', "") == 1.0

    def test_markdown_fence_invalid_schema(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        assert scorer('```json\n{"age": 30}\n```', "") == 0.5
