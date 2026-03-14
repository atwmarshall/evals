from __future__ import annotations

import re

import pytest

from evals.scorers.regex_scorer import MultiRegexScorer, RegexScorer


class TestRegexScorer:
    def test_match(self):
        assert RegexScorer(r"\d+").score("answer is 42", "") == 1.0

    def test_no_match(self):
        assert RegexScorer(r"\d+").score("no numbers here", "") == 0.0

    def test_partial_match_counts(self):
        # search() not fullmatch — partial match in string returns 1.0
        assert RegexScorer(r"hello").score("say hello world", "") == 1.0

    def test_empty_completion(self):
        assert RegexScorer(r"\d+").score("", "") == 0.0

    def test_empty_pattern_matches_everything(self):
        assert RegexScorer(r"").score("anything", "") == 1.0

    def test_empty_pattern_matches_empty(self):
        assert RegexScorer(r"").score("", "") == 1.0

    def test_case_sensitive_by_default(self):
        assert RegexScorer(r"Hello").score("hello", "") == 0.0

    def test_case_insensitive_flag(self):
        assert RegexScorer(r"(?i)hello").score("HELLO", "") == 1.0

    def test_invalid_pattern_raises(self):
        with pytest.raises(re.error):
            RegexScorer(r"[invalid")

    def test_expected_arg_ignored(self):
        # expected is not used by RegexScorer
        assert RegexScorer(r"yes").score("yes", "no") == 1.0
        assert RegexScorer(r"yes").score("no", "yes") == 0.0

    def test_multiline_anchor(self):
        assert RegexScorer(r"(?m)^bar$").score("foo\nbar\nbaz", "") == 1.0


class TestMultiRegexScorer:
    def test_empty_patterns_returns_zero(self):
        assert MultiRegexScorer([]).score("anything", "") == 0.0

    def test_all_match(self):
        assert MultiRegexScorer([r"foo", r"bar"]).score("foobar", "") == 1.0

    def test_none_match(self):
        assert MultiRegexScorer([r"foo", r"bar"]).score("baz", "") == 0.0

    def test_partial_match(self):
        assert MultiRegexScorer([r"foo", r"bar"]).score("foo only", "") == 0.5

    def test_single_pattern_match(self):
        assert MultiRegexScorer([r"\d+"]).score("42", "") == 1.0

    def test_single_pattern_no_match(self):
        assert MultiRegexScorer([r"\d+"]).score("no digits", "") == 0.0

    def test_duplicate_patterns(self):
        # each pattern counted separately, duplicates both match → 2/2 = 1.0
        assert MultiRegexScorer([r"a", r"a"]).score("a", "") == 1.0

    def test_three_patterns_one_third(self):
        result = MultiRegexScorer([r"foo", r"bar", r"baz"]).score("foo only", "")
        assert abs(result - 1 / 3) < 1e-9
