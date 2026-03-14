from __future__ import annotations

import json
import logging

import jsonschema

logger = logging.getLogger(__name__)


class JSONSchemaScorer:
    def __init__(self, schema: dict) -> None:
        self._schema = schema

    def score(self, completion: str, expected: str) -> float:
        try:
            parsed = json.loads(completion.strip())
        except json.JSONDecodeError:
            return 0.0

        try:
            jsonschema.validate(parsed, self._schema)
            return 1.0
        except jsonschema.ValidationError as e:
            logger.debug("schema validation failed: %s", e.message)
            return 0.5
