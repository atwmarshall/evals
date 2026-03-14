# CLI defaults — when to have them

## The rule
Defaults are appropriate when there's a sensible domain-specific value you'd
reach for 90% of the time. They're inappropriate when a default would silently
produce misleading results.

## Applied to this project

| Arg | Default? | Reason |
|---|---|---|
| `--criteria` (judge) | Yes — generic rubric | Better than erroring; signals expected format |
| `--schema` (schema scorer) | Yes — `EXTRACTION_SCHEMA` | You own the dataset, you know the shape |
| `--pattern` (regex) | **No** | A regex without a pattern is meaningless. Improve the error message instead: `--pattern is required (e.g. --pattern '\d{4}-\d{2}-\d{2}')` |

## EvalConfig and env vars
`EvalConfig()` does **not** automatically pick up `DEFAULT_MODEL` from the environment.
The default is hardcoded. To read env vars, call `EvalConfig.from_env()` explicitly
or read the env var in the CLI before constructing config.
Don't document it as automatic when it isn't.

## Scorer registry design
A plain dict `{"exact": exact_match}` works for zero-arg scorers but breaks
for parameterised scorers like `LLMJudge(criteria=...)` and `RegexScorer(pattern=...)`.
Use a `build_scorer(args)` factory function — one case per scorer, args injected explicitly.
```python
def build_scorer(args):
    if args.scorer == "exact":
        return exact_match
    elif args.scorer == "judge":
        return LLMJudgeScorer(criteria=args.criteria or DEFAULT_CRITERIA)
```
