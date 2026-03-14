from __future__ import annotations

import re


class RegexScorer:
    """Scorer that returns 1.0 if a regex pattern is found in the completion.

    `expected` is not used — this scorer checks format/content presence, not
    equality against a known answer.

    Pattern matching is case-insensitive by default (flags=re.IGNORECASE).
    Pass flags=0 to get case-sensitive matching.
    """

    def __init__(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        self._pattern = re.compile(pattern, flags)

    def __call__(self, completion: str, expected: str) -> float:
        return 1.0 if self._pattern.search(completion) else 0.0


class MultiRegexScorer:
    """Scorer that returns the fraction of patterns found in the completion (0.0–1.0).

    `expected` is not used — this scorer checks content presence across multiple
    required patterns, e.g. verifying a response covers several required points.

    Returns 0.0 for an empty pattern list. An empty list is almost certainly a
    caller mistake — consider raising ValueError if you prefer a hard failure.

    Pattern matching is case-insensitive by default (flags=re.IGNORECASE).
    """

    def __init__(self, patterns: list[str], flags: int = re.IGNORECASE) -> None:
        self._patterns = [re.compile(p, flags) for p in patterns]

    def __call__(self, completion: str, expected: str) -> float:
        if not self._patterns:
            # Empty pattern list: no criteria to check, return 0.0 rather than
            # 1.0 (vacuous truth) to avoid silently masking misconfiguration.
            return 0.0
        matched = sum(1 for p in self._patterns if p.search(completion))
        return matched / len(self._patterns)
