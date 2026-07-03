"""Average >= 2 benchmark runs into mean +/- stdev per metric.

Single-run deltas on this benchmark are noise (answer generation is not
seeded). Run the benchmark >= 3 times, then feed the run directories here to
get a mean +/- stdev comparison against the Khora reference baseline - that is
what lets a real regression be told apart from run-to-run variance.

Usage:
    python scripts/aggregate_runs.py results/<id1> results/<id2> results/<id3>
    python scripts/aggregate_runs.py results/<id1>/report.json ... --out agg.md

Each argument is either a run directory (containing report.json) or a
report.json path directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from khora_graphrag_bench.reporters._reference import KHORA_BASELINE
from khora_graphrag_bench.reporters.aggregate import aggregate_runs, format_comparison_markdown


def _load_result(path: Path) -> dict:
    """Load the ``result`` dict from a run dir or a report.json path."""
    if path.is_dir():
        path = path / "report.json"
    payload = json.loads(path.read_text())
    return payload.get("result", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Average >= 2 benchmark runs (mean +/- stdev per metric).")
    parser.add_argument("runs", nargs="+", help="Run dirs (results/<id>) or report.json paths.")
    parser.add_argument("--out", type=Path, help="Write the Markdown table here instead of stdout.")
    args = parser.parse_args(argv)

    if len(args.runs) < 3:
        print(
            f"warning: averaging {len(args.runs)} run(s); >= 3 is recommended before treating "
            "deltas as real (single-run spread is ~1pt on mean_answer_score).",
            file=sys.stderr,
        )

    results = [_load_result(Path(p)) for p in args.runs]
    agg = aggregate_runs(results)
    table = format_comparison_markdown(agg, reference=KHORA_BASELINE)
    doc = f"# Aggregated over {len(results)} runs\n\n{table}\n"

    if args.out:
        args.out.write_text(doc)
        print(f"wrote {args.out}")
    else:
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
