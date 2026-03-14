from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an impartial evaluator. Score the following answer using the criteria below.

Criteria:
{criteria}

Question:
{question}

Answer:
{answer}

Respond with a JSON object with two fields:
- "score": an integer from 1 to {scale} (1 = worst, {scale} = best)
- "reasoning": one sentence explaining your score

Respond with JSON only. No other text."""


@dataclass
class LLMJudge:
    criteria: str
    scale: int = 5
    model: str = field(default_factory=lambda: os.environ.get("JUDGE_MODEL", "llama3.2:3b"))
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results")))
    _session_ts: str = field(init=False)
    _trace_dir: Path = field(init=False)
    _client: ollama.Client = field(init=False)

    def __post_init__(self) -> None:
        self._session_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        self._trace_dir = self.results_dir / "judge_traces" / self._session_ts
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def score(self, completion: str, expected: str) -> float | None:
        # expected is the rubric/criteria description for judge datasets
        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            question=expected,
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
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
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
