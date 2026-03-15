from __future__ import annotations

import pytest

from evals.core import RunResult, Sample, ScorerContext
from evals.reporters import Reporter
from evals.scorers.schema import JSONSchemaScorer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SAMPLE = Sample(id="x", input="q", expected="e", metadata={})


def _make_result(metadata: dict, score: float | None = 1.0) -> RunResult:
    return RunResult(
        sample=_SAMPLE,
        completion="c",
        score=score,
        latency_ms=10,
        error=None,
        metadata=metadata,
    )


_reporter = Reporter()

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# _summarise — format_status rates
# ---------------------------------------------------------------------------

class TestSummariseFormatRates:
    def test_all_clean(self):
        results = [_make_result({"format_status": "clean"}) for _ in range(4)]
        s = _reporter._summarise(results)
        assert s["format_pass_rate"] == 1.0
        assert s["clean_rate"] == 1.0
        assert s["repair_failure_rate"] == 0.0

    def test_mixed_statuses(self):
        results = [
            _make_result({"format_status": "clean"}),
            _make_result({"format_status": "clean"}),
            _make_result({"format_status": "repaired"}),
            _make_result({"format_status": "repair_failed"}),
        ]
        s = _reporter._summarise(results)
        assert s["clean_rate"] == pytest.approx(0.5)
        assert s["format_pass_rate"] == pytest.approx(0.75)
        assert s["repair_failure_rate"] == pytest.approx(0.25)

    def test_no_format_status_keys(self):
        results = [_make_result({}) for _ in range(3)]
        s = _reporter._summarise(results)
        assert s["clean_rate"] is None
        assert s["format_pass_rate"] is None
        assert s["repair_failure_rate"] is None

    def test_rates_sum_to_one(self):
        results = [
            _make_result({"format_status": "clean"}),
            _make_result({"format_status": "repaired"}),
            _make_result({"format_status": "repair_failed"}),
        ]
        s = _reporter._summarise(results)
        total = s["clean_rate"] + (s["format_pass_rate"] - s["clean_rate"]) + s["repair_failure_rate"]
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _summarise — judge_rate from tier_used
# ---------------------------------------------------------------------------

class TestSummariseJudgeRate:
    def test_judge_rate_all_judge(self):
        results = [_make_result({"tier_used": "judge"}) for _ in range(3)]
        s = _reporter._summarise(results)
        assert s["judge_rate"] == pytest.approx(1.0)

    def test_judge_rate_mixed(self):
        results = [
            _make_result({"tier_used": "fast"}),
            _make_result({"tier_used": "fast"}),
            _make_result({"tier_used": "judge"}),
        ]
        s = _reporter._summarise(results)
        assert s["judge_rate"] == pytest.approx(1 / 3)

    def test_judge_rate_none_when_no_tier(self):
        results = [_make_result({}) for _ in range(3)]
        s = _reporter._summarise(results)
        assert s["judge_rate"] is None


# ---------------------------------------------------------------------------
# JSONSchemaScorer — writes format_status to ctx.metadata_out
# ---------------------------------------------------------------------------

class TestSchemaFormatStatus:
    def _ctx(self) -> ScorerContext:
        return ScorerContext()

    def test_clean_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        ctx = self._ctx()
        scorer('{"name": "Alice"}', "", ctx)
        assert ctx.metadata_out["format_status"] == "clean"

    def test_repaired_json(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        ctx = self._ctx()
        # truncated JSON that can be repaired
        scorer('{"name": "Al', "", ctx)
        assert ctx.metadata_out["format_status"] == "repaired"

    def test_repair_failed(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        ctx = self._ctx()
        scorer("not json at all", "", ctx)
        assert ctx.metadata_out["format_status"] == "repair_failed"

    def test_schema_fail_still_clean(self):
        scorer = JSONSchemaScorer(_SIMPLE_SCHEMA)
        ctx = self._ctx()
        # valid JSON but fails schema (missing required "name")
        scorer('{"age": 30}', "", ctx)
        assert ctx.metadata_out["format_status"] == "clean"
