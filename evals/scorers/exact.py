from __future__ import annotations

import re


def exact_match(completion: str, expected: str) -> float:
    """Return 1.0 if completion equals expected after stripping whitespace, else 0.0.

    Case-sensitive. Use normalised_match for case- and punctuation-insensitive comparison.
    """
    return 1.0 if completion.strip() == expected.strip() else 0.0


def normalised_match(completion: str, expected: str) -> float:
    """Return 1.0 if completion equals expected after normalisation, else 0.0.

    Normalisation: lowercase, strip punctuation (keeps word chars and whitespace),
    collapse runs of whitespace to a single space.

    Note: hyphens and other punctuation are stripped, so "2024-03-14" and "20240314"
    normalise identically. Underscores are preserved (they are word characters).
    """
    def normalise(s: str) -> str:
        s = re.sub(r'[^\w\s]', '', s).lower()
        return ' '.join(s.split())

    return 1.0 if normalise(completion) == normalise(expected) else 0.0
