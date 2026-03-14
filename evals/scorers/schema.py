from __future__ import annotations

import json
import logging
import re

import jsonschema

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _extract_json(text: str) -> str:
    """Strip markdown code fences and surrounding whitespace."""
    return _FENCE_RE.sub("", text).strip()


class JSONSchemaScorer:
    """Scorer that validates a completion as JSON conforming to a schema.

    Partial credit logic:
      - 0.0  — completion is not valid JSON (or empty)
      - 0.5  — valid JSON but fails schema validation
      - 1.0  — valid JSON and passes schema validation

    `expected` is not used — the schema passed at construction time defines
    what a correct response looks like.

    Handles completions wrapped in markdown code fences (```json ... ```).
    """

    def __init__(self, schema: dict) -> None:
        self._schema = schema

    def __call__(self, completion: str, expected: str) -> float:
        try:
            parsed = json.loads(_extract_json(completion))
        except json.JSONDecodeError:
            return 0.0

        try:
            jsonschema.validate(parsed, self._schema)
            return 1.0
        except jsonschema.ValidationError as e:
            logger.debug("schema validation failed: %s", e.message)
            return 0.5
