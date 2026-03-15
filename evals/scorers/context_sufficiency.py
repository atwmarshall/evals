from __future__ import annotations

import logging
import os

import ollama

from evals.core import DatasetScorer, ScorerContext

logger = logging.getLogger(__name__)

_SUFFICIENCY_PROMPT = """\
Does the following context contain enough information to answer this question,
even if the answer would use different words than the context?
Answer only YES or NO.

Context: {context}
Expected answer: {expected}"""


class ContextSufficiencyScorer(DatasetScorer):
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def __call__(self, expected: str, ctx: ScorerContext) -> float | None:
        context = ctx.metadata.get("context")
        if not context:
            return 0.0

        chunks = [context] if isinstance(context, str) else list(context)
        if not chunks:
            return 0.0

        prompt = _SUFFICIENCY_PROMPT.format(
            context="\n".join(chunks), expected=expected
        )

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw = response.message.content or ""
        except Exception as e:
            logger.error("context_sufficiency api error: %s", e)
            return None

        words = raw.strip().upper().split()
        first = words[0].rstrip(".,!?:;") if words else ""
        if first == "YES":
            ctx.metadata_out["sufficiency_reasoning"] = raw.strip()
            ctx.metadata_out["context_sufficiency_format_status"] = "clean"
            return 1.0
        elif first == "NO":
            ctx.metadata_out["sufficiency_reasoning"] = raw.strip()
            ctx.metadata_out["context_sufficiency_format_status"] = "clean"
            return 0.0
        else:
            logger.warning("context_sufficiency unexpected response: %r", raw)
            ctx.metadata_out["sufficiency_reasoning"] = raw.strip()
            ctx.metadata_out["context_sufficiency_format_status"] = "repair_failed"
            return None
