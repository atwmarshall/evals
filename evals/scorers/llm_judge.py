from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an impartial evaluator. Score the following answer using the criteria below.

Criteria:
{criteria}

Answer:
{answer}

Respond with a JSON object with two fields:
- "score": an integer from 1 to {scale} (1 = worst, {scale} = best)
- "reasoning": one sentence explaining your score

Respond with JSON only. No other text."""


class LLMJudgeScorer:
    """LLM-based scorer that evaluates completions against a rubric using a judge model.

    Internally uses a 1–scale integer rating, normalised to 0.0–1.0 via
    (score - 1) / (scale - 1). So score=1 → 0.0, score=scale → 1.0.

    Unlike other scorers, __call__ returns float | None. None signals a judge
    failure (API error or unparseable response) and is stored in RunResult.error.
    This deliberately breaks the (str, str) -> float contract so that parse
    failures are distinguishable from genuine low scores in the reporter.

    The `expected` argument to __call__ is unused — the rubric is encoded in
    `criteria` at construction time. The parameter exists only to satisfy the
    scorer interface.

    Traces (prompt, raw response, parsed score, error) are written as JSON to
    results/judge_traces/{session_timestamp}/{trace_timestamp}.json for debugging.
    """

    def __init__(
        self,
        criteria: str,
        scale: int = 5,
        model: str | None = None,
        results_dir: Path | None = None,
    ) -> None:
        self.criteria = criteria
        self.scale = scale
        self.model = model or os.environ.get("JUDGE_MODEL", "llama3.2:3b")
        self.results_dir = results_dir or Path(os.environ.get("RESULTS_DIR", "results"))
        self._session_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._trace_dir = self.results_dir / "judge_traces" / self._session_ts
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def __call__(self, completion: str, expected: str) -> float | None:  # noqa: ARG002
        # expected is unused — the rubric lives in self.criteria
        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            answer=completion,
            scale=self.scale,
        )

        raw_response: str = ""
        parsed_score: int | None = None
        final_score: float | None = None
        error: str | None = None

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw_response = response.message.content or ""
            parsed_score, error = self._parse_response(raw_response)
            if parsed_score is not None:
                final_score = (parsed_score - 1) / (self.scale - 1)
        except Exception as e:
            error = str(e)
            logger.error("llm_judge api error: %s", e)

        self._write_trace(prompt, raw_response, parsed_score, final_score, error)

        if error and final_score is None:
            return None
        return final_score

    def _parse_response(self, raw: str) -> tuple[int | None, str | None]:
        text = _strip_fences(raw)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None, f"judge response not valid JSON: {raw!r}"

        if "score" not in obj:
            return None, f"judge response missing 'score' field: {obj!r}"

        try:
            score = int(obj["score"])
        except (TypeError, ValueError):
            return None, f"judge 'score' not an integer: {obj['score']!r}"

        if not (1 <= score <= self.scale):
            return None, f"judge score {score} out of range 1–{self.scale}"

        return score, None

    def _write_trace(
        self,
        prompt: str,
        raw_response: str,
        parsed_score: int | None,
        final_score: float | None,
        error: str | None,
    ) -> None:
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        trace_ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        trace_path = self._trace_dir / f"{trace_ts}.json"
        trace = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "model": self.model,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_score": parsed_score,
            "final_score": final_score,
            "error": error,
        }
        trace_path.write_text(json.dumps(trace, indent=2))
        logger.debug("judge trace written to %s", trace_path)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()
