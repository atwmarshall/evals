from __future__ import annotations

import pytest

# _parse_response is tested by instantiating LLMJudge with a tmp trace dir
# to avoid touching the filesystem in an unexpected place.


@pytest.fixture
def judge(tmp_path):
    from evals.scorers.llm_judge import LLMJudge
    return LLMJudge(criteria="Is the answer correct?", scale=5, results_dir=tmp_path)


class TestParseResponse:
    def test_valid_response(self, judge):
        score, err = judge._parse_response('{"score": 3, "reasoning": "ok"}')
        assert score == 3
        assert err is None

    def test_minimum_score(self, judge):
        score, err = judge._parse_response('{"score": 1, "reasoning": "bad"}')
        assert score == 1
        assert err is None

    def test_maximum_score(self, judge):
        score, err = judge._parse_response('{"score": 5, "reasoning": "great"}')
        assert score == 5
        assert err is None

    def test_markdown_code_fence(self, judge):
        raw = '```json\n{"score": 4, "reasoning": "good"}\n```'
        score, err = judge._parse_response(raw)
        assert score == 4
        assert err is None

    def test_markdown_fence_no_language(self, judge):
        raw = '```\n{"score": 2, "reasoning": "ok"}\n```'
        score, err = judge._parse_response(raw)
        assert score == 2
        assert err is None

    def test_not_json(self, judge):
        score, err = judge._parse_response("I think the score is 3 out of 5.")
        assert score is None
        assert err is not None

    def test_missing_score_field(self, judge):
        score, err = judge._parse_response('{"reasoning": "looks good"}')
        assert score is None
        assert err is not None
        assert "score" in err

    def test_score_is_numeric_string(self, judge):
        # int("3") succeeds in Python, so "3" is accepted
        score, err = judge._parse_response('{"score": "3", "reasoning": "ok"}')
        assert score == 3
        assert err is None

    def test_score_is_float(self, judge):
        # floats are not ints — int("3.5") raises, int(3.5) truncates
        # json parses 3.5 as float; int(3.5) = 3 which IS valid — document the behaviour
        score, err = judge._parse_response('{"score": 3.5, "reasoning": "ok"}')
        # int(3.5) == 3, which is in range — acceptable truncation
        assert score == 3
        assert err is None

    def test_score_below_range(self, judge):
        score, err = judge._parse_response('{"score": 0, "reasoning": "terrible"}')
        assert score is None
        assert err is not None

    def test_score_above_range(self, judge):
        score, err = judge._parse_response('{"score": 6, "reasoning": "amazing"}')
        assert score is None
        assert err is not None

    def test_empty_string(self, judge):
        score, err = judge._parse_response("")
        assert score is None
        assert err is not None

    def test_extra_fields_ignored(self, judge):
        score, err = judge._parse_response('{"score": 3, "reasoning": "ok", "extra": "ignored"}')
        assert score == 3
        assert err is None
