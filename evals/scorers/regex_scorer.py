from __future__ import annotations

import re


class RegexScorer:
    def __init__(self, pattern: str) -> None:
        self._re = re.compile(pattern)

    def score(self, completion: str, expected: str) -> float:
        return 1.0 if self._re.search(completion) else 0.0


class MultiRegexScorer:
    def __init__(self, patterns: list[str]) -> None:
        self._patterns = [re.compile(p) for p in patterns]

    def score(self, completion: str, expected: str) -> float:
        if not self._patterns:
            return 0.0
        matched = sum(1 for p in self._patterns if p.search(completion))
        return matched / len(self._patterns)
