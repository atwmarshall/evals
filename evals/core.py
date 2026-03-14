from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    id: str
    input: str
    expected: str
    metadata: dict


@dataclass
class Dataset:
    samples: list[Sample]

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: int | None = None) -> Dataset:
        path = Path(path)
        samples: list[Sample] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                metadata = dict(obj.get("metadata") or {})
                for key in obj:
                    if key not in {"id", "input", "expected", "metadata"}:
                        metadata[key] = obj[key]
                samples.append(Sample(
                    id=obj["id"],
                    input=obj["input"],
                    expected=obj["expected"],
                    metadata=metadata,
                ))
        if limit is not None:
            samples = samples[:limit]
        return cls(samples=samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


@dataclass
class RunResult:
    sample: Sample
    completion: str | None
    score: float | None
    latency_ms: int
    error: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    model: str = "llama3.2"
    max_tokens: int = 1024
    temperature: float = 0.0
    system_prompt: str = ""
    max_retries: int = 3
    timeout_seconds: int = 30
