from __future__ import annotations


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to close a truncated JSON string.

    Handles: unterminated strings, missing closing } or ].
    Returns the repaired string if something was added, else None.
    """
    text = text.rstrip().rstrip(",")  # trailing comma before EOF

    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    closers = ('"' if in_string else "") + "".join(reversed(stack))
    if not closers:
        return None  # nothing truncated — don't touch it
    return text + closers
