from __future__ import annotations

import dataclasses
from collections.abc import Callable

from evals.core import Dataset, EvalConfig, RunResult, ScorerContext
from evals.runner import Runner


class BenchmarkRunner:
    """Runs a dataset + scorer across multiple model IDs sequentially.

    Sequential (not parallel) to keep latency measurements comparable —
    parallel runs share Ollama resources and would contaminate p50/p95 numbers.
    """

    def run(
        self,
        dataset: Dataset,
        scorer: Callable[[str, str, ScorerContext], float | None],
        model_ids: list[str],
        base_config: EvalConfig | None = None,
    ) -> list[tuple[str, list[RunResult]]]:
        base = base_config or EvalConfig()
        model_results: list[tuple[str, list[RunResult]]] = []

        for model_id in model_ids:
            if hasattr(scorer, "set_evaluated_model"):
                scorer.set_evaluated_model(model_id)
            config = dataclasses.replace(base, model=model_id)
            results = Runner().run(dataset, scorer, config)
            model_results.append((model_id, results))

        return model_results
