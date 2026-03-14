from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evals.core import Dataset, EvalConfig
from evals.reporters import Reporter
from evals.runner import Runner
from evals.scorers.exact import exact_match, normalised_match


def build_scorer(args: argparse.Namespace) -> Callable[[str, str], float]:
    match args.scorer:
        case "exact":
            return exact_match
        case "normalised":
            return normalised_match
        case _:
            sys.exit(f"Unknown scorer: {args.scorer!r}. Choose from: exact, normalised")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run an LLM eval")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help="Scorer name (exact, normalised)")
    parser.add_argument("--model", default=None, help="Model ID (overrides DEFAULT_MODEL env var)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run")
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR env var)")
    args = parser.parse_args()

    scorer = build_scorer(args)

    ds = Dataset.from_jsonl(args.dataset, limit=args.limit)

    config = EvalConfig(model=args.model) if args.model else EvalConfig()

    wrapped = tqdm(ds, total=len(ds), desc="Running eval")
    results = Runner().run(wrapped, scorer, config)

    reporter_kwargs = {}
    if args.output:
        reporter_kwargs["results_dir"] = Path(args.output)
    reporter = Reporter(**reporter_kwargs)

    dataset_name = Path(args.dataset).stem
    scorer_name = args.scorer

    output, path = reporter.report(results, dataset_name, scorer_name)
    print(output)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
