from __future__ import annotations

"""inspect.py — read benchmark/run/sample results without opening files manually.

Usage:
  inspect.py <path>              auto-detect type (benchmark dir / run dir / .jsonl)
  inspect.py <path> --id c3-001  show one sample in full (jsonl or benchmark model jsonl)
  inspect.py <path> --verbose    include full completions in error listings
"""

import argparse
import json
import sys
from pathlib import Path

from tabulate import tabulate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _short(text: str | None, n: int = 200) -> str:
    if text is None:
        return "(none)"
    text = text.replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def _is_failure(r: dict) -> bool:
    s = r.get("score")
    return s is None or s < 1.0


def _error_type(row: dict) -> str:
    score = row.get("score")
    if score is not None:
        return "pass" if score >= 1.0 else "fail"
    err = row.get("error") or ""
    if "timed out" in err:
        return "timeout"
    if "scorer returned None" in err:
        return "parse_failure"
    if err:
        return "api_error"
    return "pass"


def _find_trace(results_dir: Path, date: str, model_id: str, sample_id: str) -> dict | None:
    """Locate a judge trace JSON for a given model+sample under results/judge_traces/{date}/."""
    traces_root = results_dir / "judge_traces" / date
    if not traces_root.exists():
        return None
    # session dirs are named {time}_{model_id} — match by suffix
    suffix = f"_{model_id}"
    for session_dir in sorted(traces_root.iterdir()):
        if session_dir.name.endswith(suffix):
            trace_file = session_dir / f"{sample_id}.json"
            if trace_file.exists():
                return json.loads(trace_file.read_text())
    return None


# ---------------------------------------------------------------------------
# benchmark mode
# ---------------------------------------------------------------------------

def inspect_benchmark(bench_dir: Path, verbose: bool, sample_id: str | None = None, failures_only: bool = False) -> None:
    meta = json.loads((bench_dir / "benchmark.json").read_text())
    date = bench_dir.parent.name  # results/benchmarks/{date}/...
    results_dir = bench_dir.parent.parent.parent  # results/

    no_traces_note = ""
    if meta.get("scorer") not in ("judge", "cascade"):
        no_traces_note = "  (no judge traces — scorer is not judge/cascade)"
    print(f"BENCHMARK  dataset={meta['dataset']}  scorer={meta['scorer']}  {meta['timestamp']}{no_traces_note}")
    print(f"dir: {bench_dir}\n")

    # --- --id: show one sample across all models ---
    if sample_id:
        _benchmark_sample(bench_dir, meta, results_dir, date, sample_id, verbose, failures_only=failures_only)
        return

    # --- comparison table from benchmark.json (suppressed with --failures-only) ---
    if not failures_only:
        table_rows = []
        for model_id, s in meta["models"].items():
            mean = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
            p95_str = f"{s['p95_latency_ms']}ms"
            if s.get("n", 99) < 20:
                p95_str += f" (n={s['n']}⚠)"
            table_rows.append([
                model_id,
                mean,
                f"{s['p50_latency_ms']}ms",
                p95_str,
                f"{s['error_rate']:.1%}",
                s["api_errors"] + s["parse_failures"],
            ])
        print(tabulate(table_rows,
                       headers=["model", "mean_score", "p50", "p95", "error_rate", "n_errors"],
                       tablefmt="simple"))

    # --- error breakdown ---
    any_errors = False
    for jsonl_path in sorted(bench_dir.glob("*.jsonl")):
        # reverse-engineer model_id from filename: replace _ with : for the colon, but
        # we can't perfectly invert sanitisation — use benchmark.json model keys instead
        rows = _load_jsonl(jsonl_path)
        failures = [r for r in rows if _error_type(r) not in ("pass", "fail")]
        if not failures:
            continue

        if not any_errors:
            print("\n── ERROR BREAKDOWN " + "─" * 60)
            any_errors = True

        # match filename stem back to a model id from benchmark.json
        stem = jsonl_path.stem
        model_id = next(
            (m for m in meta["models"] if stem == m.replace(":", "_").replace("/", "_")),
            stem,
        )
        print(f"\n[{model_id}]  {len(failures)}/{len(rows)} failed")

        for r in failures:
            etype = _error_type(r)
            print(f"  {r['id']}  type={etype}  latency={r['latency_ms']}ms")

            if etype == "timeout":
                print(f"    error: {r.get('error', '')}")

            elif etype == "parse_failure":
                # try to find the trace
                trace = _find_trace(results_dir, date, model_id, r["id"])
                if trace:
                    raw = trace.get("raw_response", "")
                    print(f"    judge raw: {_short(raw, 300 if verbose else 150)}")
                    if trace.get("error"):
                        print(f"    parse error: {trace['error'][:120]}")
                else:
                    print(f"    (no trace found — check results/judge_traces/{date}/)")
                if verbose and r.get("completion"):
                    print(f"    completion: {_short(r['completion'], 400)}")

            elif etype == "api_error":
                print(f"    error: {r.get('error', '')}")

    if not any_errors:
        scorer = meta.get("scorer", "")
        if scorer not in ("judge", "cascade"):
            print(f"\nNo errors.  (scorer={scorer!r} — no judge traces generated)")
        else:
            print("\nNo errors.")


# ---------------------------------------------------------------------------
# benchmark --id helper: one sample across all models
# ---------------------------------------------------------------------------

def _benchmark_sample(
    bench_dir: Path,
    meta: dict,
    results_dir: Path,
    date: str,
    sample_id: str,
    verbose: bool,
    failures_only: bool = False,
) -> None:
    print(f"── sample {sample_id} across all models ──\n")
    found_any = False
    expected_shown = False
    for jsonl_path in sorted(bench_dir.glob("*.jsonl")):
        rows = _load_jsonl(jsonl_path)
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            continue
        r = matches[0]

        if failures_only and not _is_failure(r):
            continue

        found_any = True
        # show expected once at the top (same for all models)
        if not expected_shown and r.get("expected") is not None:
            print(f"expected:  {r['expected']}\n")
            expected_shown = True

        stem = jsonl_path.stem
        model_id = next(
            (m for m in meta["models"] if stem == m.replace(":", "_").replace("/", "_")),
            stem,
        )
        etype = _error_type(r)
        score_str = f"{r['score']:.3f}" if r.get("score") is not None else "—"
        print(f"[{model_id}]  score={score_str}  latency={r['latency_ms']}ms  type={etype}")

        if etype == "parse_failure":
            trace = _find_trace(results_dir, date, model_id, sample_id)
            if trace:
                print(f"  judge raw: {_short(trace.get('raw_response', ''), 200)}")

        if etype == "timeout":
            print(f"  error: {r.get('error')}")

        if verbose and r.get("completion"):
            print(f"  completion: {_short(r['completion'], 600)}")
        print()

    if not found_any:
        if failures_only:
            print(f"No failures found for sample {sample_id!r} across all models.")
        else:
            print(f"Sample {sample_id!r} not found in any model jsonl under {bench_dir}")


# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------

def inspect_run(run_dir: Path, verbose: bool, sample_id: str | None = None, failures_only: bool = False) -> None:
    meta = json.loads((run_dir / "run.json").read_text())
    rows = _load_jsonl(run_dir / "samples.jsonl")

    print(f"RUN  model={meta['model']}  dataset={meta['dataset']}  scorer={meta['scorer']}  {meta['timestamp']}")
    s = meta["summary"]
    mean = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "—"
    print(f"     mean_score={mean}  p50={s['p50_latency_ms']}ms  errors={s['api_errors']+s['parse_failures']}/{s['n']}\n")

    if sample_id:
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            print(f"Sample {sample_id!r} not found.")
            sys.exit(1)
        r = matches[0]
        print(f"id={r['id']}  score={r.get('score')}  latency={r.get('latency_ms')}ms  type={_error_type(r)}")
        if r.get("expected") is not None:
            print(f"expected:  {r['expected']}")
        print(f"error: {r.get('error')}")
        print(f"\ncompletion:\n{r.get('completion') or '(none)'}")
        return

    display_rows = [r for r in rows if _is_failure(r)] if failures_only else rows
    table = [
        [
            r["id"],
            f"{r['score']:.2f}" if r["score"] is not None else "—",
            r["latency_ms"],
            _error_type(r),
            _short(r.get("error") or "", 60),
        ]
        for r in display_rows
    ]
    if failures_only:
        print(f"(failures only: {len(display_rows)}/{len(rows)})\n")
    print(tabulate(table, headers=["id", "score", "latency_ms", "type", "error"], tablefmt="simple"))

    failures = [r for r in rows if _error_type(r) != "pass"]
    if failures and verbose:
        print("\n── COMPLETIONS FOR FAILURES " + "─" * 50)
        for r in failures:
            print(f"\n[{r['id']}]")
            print(f"  completion: {_short(r.get('completion'), 400)}")
            print(f"  error:      {r.get('error')}")


# ---------------------------------------------------------------------------
# jsonl/sample mode
# ---------------------------------------------------------------------------

def inspect_jsonl(path: Path, sample_id: str | None, verbose: bool, failures_only: bool = False) -> None:
    rows = _load_jsonl(path)

    if sample_id:
        matches = [r for r in rows if r.get("id") == sample_id]
        if not matches:
            print(f"Sample {sample_id!r} not found in {path}")
            sys.exit(1)
        r = matches[0]
        print(f"id={r['id']}  score={r.get('score')}  latency={r.get('latency_ms')}ms")
        if r.get("expected") is not None:
            print(f"expected:  {r['expected']}")
        print(f"error: {r.get('error')}")
        print(f"\ncompletion:\n{r.get('completion') or '(none)'}")
        return

    display_rows = [r for r in rows if _is_failure(r)] if failures_only else rows
    table = [
        [
            r.get("id"),
            f"{r['score']:.2f}" if r.get("score") is not None else "—",
            r.get("latency_ms"),
            _error_type(r),
            _short(r.get("error") or "", 60),
        ]
        for r in display_rows
    ]
    label = f"  (failures only: {len(display_rows)}/{len(rows)})" if failures_only else f"  ({len(rows)} samples)"
    print(f"{path}{label}\n")
    print(tabulate(table, headers=["id", "score", "latency_ms", "type", "error"], tablefmt="simple"))

    if verbose:
        print()
        for r in rows:
            print(f"── {r['id']}  score={r.get('score')} ──")
            print(r.get("completion") or "(none)")
            print()


# ---------------------------------------------------------------------------
# traces mode
# ---------------------------------------------------------------------------

def inspect_traces(traces_dir: Path, verbose: bool, failures_only: bool = False) -> None:
    """Summarise judge traces under a session directory."""
    trace_files = sorted(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        sys.exit(1)

    all_rows = []
    for tf in trace_files:
        t = json.loads(tf.read_text())
        status = "ok" if t.get("final_score") is not None else "fail"
        all_rows.append((tf, t, status))

    display = [(tf, t, s) for tf, t, s in all_rows if s != "ok"] if failures_only else all_rows
    label = f"  (failures only: {len(display)}/{len(all_rows)})" if failures_only else f"  ({len(all_rows)} samples)"
    print(f"TRACES  dir={traces_dir}{label}\n")

    rows = [
        [
            t.get("sample_id", tf.stem),
            t.get("evaluated_model", "?"),
            f"{t['final_score']:.2f}" if t.get("final_score") is not None else "—",
            t.get("parsed_score"),
            s,
            _short(t.get("error") or "", 80),
        ]
        for tf, t, s in display
    ]
    print(tabulate(rows,
                   headers=["sample_id", "evaluated_model", "score", "raw_score", "status", "error"],
                   tablefmt="simple"))

    if verbose:
        for tf, t, s in display:
            if t.get("error"):
                print(f"\n── {tf.stem} ──")
                print(f"raw_response: {_short(t.get('raw_response', ''), 400)}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect eval results")
    parser.add_argument("path", help="Benchmark dir / run dir / .jsonl file / traces dir")
    parser.add_argument("--id", default=None, help="Show a specific sample by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full completions")
    parser.add_argument("--failures-only", "-f", action="store_true", help="Show only samples with score < 1.0 or score=None")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    if path.is_file() and path.suffix == ".jsonl":
        inspect_jsonl(path, args.id, args.verbose, failures_only=args.failures_only)
    elif path.is_dir() and (path / "benchmark.json").exists():
        inspect_benchmark(path, args.verbose, sample_id=args.id, failures_only=args.failures_only)
    elif path.is_dir() and (path / "run.json").exists():
        inspect_run(path, args.verbose, sample_id=args.id, failures_only=args.failures_only)
    elif path.is_dir() and any(path.glob("*.json")):
        inspect_traces(path, args.verbose, failures_only=args.failures_only)
    else:
        print(f"Cannot determine type of {path}")
        if path.is_dir():
            candidates = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob("*.json")))
            if candidates:
                print("\nDid you mean one of these?")
                for c in candidates:
                    print(f"  {c.name}/")
                sys.exit(1)
        print("Expected: benchmark dir (has benchmark.json), run dir (has run.json), .jsonl file, or traces dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
