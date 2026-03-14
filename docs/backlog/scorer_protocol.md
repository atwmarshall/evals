# Backlog: ScorerProtocol

Replace the `Callable[[str, str, ScorerContext], float | None]` type alias in `runner.py` with a `typing.Protocol`.

## What

```python
from typing import Protocol
from evals.core import ScorerContext

class ScorerProtocol(Protocol):
    def __call__(self, completion: str, expected: str, ctx: ScorerContext) -> float | None: ...
```

Use `ScorerProtocol` as the type for the `scorer` parameter in `Runner.run` and `BenchmarkRunner`.

## Why

The current `Callable[...]` annotation is hard to read and doesn't show parameter names. A Protocol makes the contract explicit, is inspectable, and lets a type checker (`mypy`, `pyright`) catch scorers with wrong signatures at definition time rather than at call sites.

## When to do it

After the scorer interface stabilises — i.e. once Challenge 7 (RAG) is done and `ScorerContext` has its final shape (`retrieved_chunks` etc.). Adding the Protocol before the interface is settled just means updating it twice.
