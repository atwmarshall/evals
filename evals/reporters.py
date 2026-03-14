from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

from evals.core import RunResult


def _sanitise_model(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


@dataclass
class Reporter:
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("RESULTS_DIR", "results")))

    def _summarise(self, results: list[RunResult]) -> dict:
        latencies = [r.latency_ms for r in results]
        scores = [r.score for r in results if r.score is not None]
        api_errors = sum(1 for r in results if r.error and r.score is None and r.completion is None)
        parse_failures = sum(1 for r in results if r.error and r.score is None and r.completion is not None)
        n = len(results)
        mean_score = statistics.mean(scores) if scores else None
        p50_latency = int(statistics.median(latencies)) if latencies else 0
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0
        total_errors = api_errors + parse_failures
        error_rate = total_errors / n if n else 0.0
        return {
            "mean_score": mean_score,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p95_low_confidence": n < 20,
            "n": n,
            "api_errors": api_errors,
            "parse_failures": parse_failures,
            "error_rate": error_rate,
        }

    def report(
        self,
        results: list[RunResult],
        dataset_name: str,
        scorer_name: str,
        model: str = "unknown",
    ) -> tuple[str, Path]:
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

        summary = self._summarise(results)
        mean_score = summary["mean_score"]
        p50_latency = summary["p50_latency_ms"]
        p95_latency = summary["p95_latency_ms"]
        n = summary["n"]

        p95_str = f"{p95_latency}ms"
        if summary["p95_low_confidence"]:
            p95_str += f" (n={n} ⚠)"

        summary_parts = [
            f"mean_score={mean_score:.3f}" if mean_score is not None else "mean_score=—",
            f"p50_latency={p50_latency}ms",
            f"p95_latency={p95_str}",
            f"api_errors={summary['api_errors']}",
            f"parse_failures={summary['parse_failures']}",
            f"error_rate={summary['error_rate']:.1%}",
        ]
        summary_str = "  ".join(summary_parts)

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        safe_model = _sanitise_model(model)
        run_dir = (
            self.results_dir / "runs" / date / f"{time_str}_{safe_model}_{dataset_name}_{scorer_name}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        run_payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "model": model,
            "timestamp": now.isoformat(timespec="seconds"),
            "summary": {k: v for k, v in summary.items() if k != "p95_low_confidence"},
        }
        (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2))

        with (run_dir / "samples.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps({
                    "id": r.sample.id,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "completion": r.completion,
                    "error": r.error,
                }) + "\n")

        return f"{table_str}\n\n{summary_str}", run_dir

    def benchmark_report(
        self,
        model_results: list[tuple[str, list[RunResult]]],
        dataset_name: str,
        scorer_name: str,
    ) -> tuple[str, Path]:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M%S")
        bench_dir = (
            self.results_dir / "benchmarks" / date / f"{time_str}_{dataset_name}_{scorer_name}"
        )
        bench_dir.mkdir(parents=True, exist_ok=True)

        summaries: dict[str, dict] = {}
        table_rows = []

        for model_id, results in model_results:
            s = self._summarise(results)
            summaries[model_id] = s

            mean_score_str = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
            p95_str = f"{s['p95_latency_ms']}ms"
            if s["p95_low_confidence"]:
                p95_str += f" (n={s['n']} ⚠)"

            table_rows.append([
                model_id,
                mean_score_str,
                f"{s['p50_latency_ms']}ms",
                p95_str,
                f"{s['error_rate']:.1%}",
            ])

            safe_model = _sanitise_model(model_id)
            with (bench_dir / f"{safe_model}.jsonl").open("w") as f:
                for r in results:
                    f.write(json.dumps({
                        "id": r.sample.id,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "completion": r.completion,
                        "error": r.error,
                    }) + "\n")

        table_str = tabulate(
            table_rows,
            headers=["model", "mean_score", "p50_latency", "p95_latency", "error_rate"],
            tablefmt="simple",
        )

        benchmark_payload = {
            "dataset": dataset_name,
            "scorer": scorer_name,
            "timestamp": now.isoformat(timespec="seconds"),
            "models": {
                model_id: {k: v for k, v in s.items() if k != "p95_low_confidence"}
                for model_id, s in summaries.items()
            },
        }
        (bench_dir / "benchmark.json").write_text(json.dumps(benchmark_payload, indent=2))

        return table_str, bench_dir
