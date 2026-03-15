# show.py: robustness result support

## Gap

`show.py` has no detection clause for `robustness.json`. Running `show.py` against a robustness result directory falls through to the `.jsonl` handler, which will display the per-perturbation JSONL files individually but not the summary or degradation table from `robustness.json`.

Contrast with sensitivity results, which have a dedicated `inspect_sensitivity()` function triggered when `sensitivity.json` is present.

## What's missing

A detection clause in `show.py`'s path-type resolver:

```python
if (path / "robustness.json").exists():
    inspect_robustness(path, args)
```

And an `inspect_robustness()` function that:

- Reads `robustness.json` and prints the per-sample degradation table and per-perturbation summary (same format as `RobustnessReporter.report()` prints to stdout during the run)
- Supports `--id <sample_id>` to show one sample's scores across all perturbation types
- Supports `--failures-only` to filter to `fragile` and `brittle` rows only (parallel to sensitivity's `--failures-only` filtering on `unstable`)
- Supports `--verbose` to show completions from the per-perturbation JSONL files

## Why it matters

Without this, inspecting a robustness result after the run requires either re-running it (wasteful, requires the model to be available) or opening `robustness.json` manually. The per-perturbation JSONL files are readable by `show.py`'s existing `.jsonl` handler, but that shows raw results without the degradation metric or verdict.

## Detection order

`show.py` currently resolves path type in this order:
1. sensitivity dir (`sensitivity.json` present)
2. benchmark dir (`benchmark.json` present)
3. run dir (`run.json` present)
4. `.jsonl` file
5. traces dir (contains `*.json`)

Robustness should be inserted at position 1 (before or alongside sensitivity):
1. robustness dir (`robustness.json` present)
2. sensitivity dir (`sensitivity.json` present)
3. ...

## Notes

- The `--failures-only` flag for robustness should filter on `verdict in {"fragile", "brittle"}`, not just `"unstable"`
- `run_config` stored in `robustness.json` does not include `validation_threshold` or `judge_model` — the display should not expect those keys
- `most_degrading_n` and `baseline_n` in `summary` should be surfaced when they differ (same n-note logic as the live report)
