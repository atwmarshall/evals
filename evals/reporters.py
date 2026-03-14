from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from evals.core import RunResult


@dataclass
class Reporter:
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results")))

    def report(self, results: list[RunResult], dataset_name: str, scorer_name: str) -> tuple[str, Path]:
        rows = [
            [
                r.sample.id,
                f"{r.score:.2f}" if r.score is not None else "—",
                r.latency_ms,
                r.error or "",
            ]
            for r in results
        ]
        table_str = tabulate(rows, headers=["id", "score", "latency_ms", "error"], tablefmt="simple")

        latencies = [r.latency_ms for r in results]
        scores = [r.score for r in results if r.score is not None]
        api_errors = sum(1 for r in results if r.error and r.score is None and r.completion is None)
        parse_failures = sum(1 for r in results if r.error and r.score is None and r.completion is not None)

        mean_score = statistics.mean(scores) if scores else 0.0
        p50_latency = int(statistics.median(latencies)) if latencies else 0
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0
        total_errors = api_errors + parse_failures
        error_rate = total_errors / len(results) if results else 0.0

        summary_parts = [
            f"mean_score={mean_score:.3f}",
            f"p50_latency={p50_latency}ms",
            f"p95_latency={p95_latency}ms",
            f"api_errors={api_errors}",
            f"parse_failures={parse_failures}",
            f"error_rate={error_rate:.1%}",
        ]
        summary_str = "  ".join(summary_parts)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.results_dir / f"{timestamp}_{dataset_name}_{scorer_name}.json"

        payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "mean_score": mean_score,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "api_errors": api_errors,
                "parse_failures": parse_failures,
                "error_rate": error_rate,
            },
            "results": [
                {
                    "id": r.sample.id,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "completion": r.completion,
                    "error": r.error,
                }
                for r in results
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2))

        return f"{table_str}\n\n{summary_str}", output_path
