"""JSON reporter — dumps the full ``BenchmarkRunResult`` to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from khora_graphrag_bench.harness.results import BenchmarkRunResult
from khora_graphrag_bench.reporters._reference import KHORA_BASELINE


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"object of type {type(value).__name__!r} not JSON-serialisable")


def write_json_report(result: BenchmarkRunResult, out_dir: str | Path) -> Path:
    """Write the full result + reference baseline to ``{out_dir}/report.json``."""
    out_path = Path(out_dir) / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "khora_reference_baseline": KHORA_BASELINE,
        "result": asdict(result),
    }
    out_path.write_text(json.dumps(payload, default=_json_default, indent=2))
    return out_path
