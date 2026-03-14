from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from evals.scorers.cascade import CascadeScorer
from evals.scorers.exact import exact_match, normalised_match
from evals.scorers.llm_judge import LLMJudgeScorer
from evals.scorers.regex import MultiRegexScorer, RegexScorer
from evals.scorers.schema import JSONSchemaScorer

SCORER_CHOICES = "exact, normalised, regex, multi-regex, schema, judge, cascade"

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


def build_scorer(
    args: argparse.Namespace,
    evaluated_model: str | None = None,
) -> Callable[[str, str], float]:
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
                evaluated_model=evaluated_model,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
        case "cascade":
            fast = normalised_match if args.fast_tier == "normalised" else exact_match
            judge = LLMJudgeScorer(
                scale=args.scale,
                evaluated_model=evaluated_model,
                **({"model": args.judge_model} if args.judge_model else {}),
            )
            return CascadeScorer(fast=fast, judge=judge, threshold=args.threshold)
        case _:
            sys.exit(f"Unknown scorer: {args.scorer!r}. Choose from: {SCORER_CHOICES}")
