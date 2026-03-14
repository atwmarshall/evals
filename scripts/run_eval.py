from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from evals.core import Dataset, EvalConfig
from evals.reporters import Reporter
from evals.runner import Runner
from evals.scorers.exact import exact_match, normalised_match
from evals.scorers.llm_judge import LLMJudgeScorer
from evals.scorers.regex import MultiRegexScorer, RegexScorer
from evals.scorers.schema import JSONSchemaScorer

SCORER_CHOICES = "exact, normalised, regex, multi-regex, schema, judge"

# Default schema for the extraction dataset (company/date/amount invoice extraction).
# Pass --schema to override with a different JSON schema file.
_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "amount": {"type": "number"},
    },
    "required": ["company", "date", "amount"],
    "additionalProperties": False,
}



def build_scorer(args: argparse.Namespace) -> Callable[[str, str], float]:
    match args.scorer:
        case "exact":
            return exact_match
        case "normalised":
            return normalised_match
        case "regex":
            if not args.pattern:
                sys.exit(
                    "--pattern is required for scorer 'regex' "
                    r"(e.g. --pattern '\d{4}-\d{2}-\d{2}')"
                )
            return RegexScorer(args.pattern)
        case "multi-regex":
            if not args.pattern:
                sys.exit(
                    "--pattern is required for scorer 'multi-regex' "
                    r"(e.g. --pattern 'company,date,amount' — comma-separated patterns)"
                )
            patterns = [p.strip() for p in args.pattern.split(",")]
            return MultiRegexScorer(patterns)
        case "schema":
            if args.schema:
                schema = json.loads(Path(args.schema).read_text())
            else:
                schema = _EXTRACTION_SCHEMA
            return JSONSchemaScorer(schema)
        case "judge":
            return LLMJudgeScorer(
                scale=args.scale,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
        case _:
            sys.exit(f"Unknown scorer: {args.scorer!r}. Choose from: {SCORER_CHOICES}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run an LLM eval")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--scorer", required=True, help=f"Scorer name ({SCORER_CHOICES})")
    parser.add_argument("--model", default=None, help="Model ID (overrides DEFAULT_MODEL env var)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run")
    parser.add_argument("--output", default=None, help="Results directory (overrides RESULTS_DIR env var)")
    # Scorer-specific args
    parser.add_argument("--pattern", default=None, help="Regex pattern(s) for regex/multi-regex scorers")
    parser.add_argument("--schema", default=None, help="Path to JSON schema file for schema scorer")
    parser.add_argument("--scale", type=int, default=5, help="Score scale for judge scorer (default: 5)")
    parser.add_argument("--judge-model", default=None, help="Model ID for judge (overrides JUDGE_MODEL env var)")
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
