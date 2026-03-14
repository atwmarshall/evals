from __future__ import annotations

import re


def exact_match(completion: str, expected: str) -> float:
    return 1.0 if completion.strip() == expected.strip() else 0.0


def normalised_match(completion: str, expected: str) -> float:
    def normalise(s: str) -> str:
        s = re.sub(r'[^\w\s]', '', s).lower()
        return ' '.join(s.split())

    return 1.0 if normalise(completion) == normalise(expected) else 0.0
