"""HTML reporter — self-contained single-page report."""

from __future__ import annotations

from pathlib import Path

from khora_graphrag_bench.harness.results import BenchmarkRunResult
from khora_graphrag_bench.reporters._reference import KHORA_BASELINE

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GraphRAG-Bench — {adapter_name}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 32px auto; padding: 0 16px; color: #222; }}
  h1 {{ margin-bottom: 8px; }}
  h2 {{ margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 24px; }}
  .meta span {{ display: inline-block; margin-right: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #fafafa; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .delta-pos {{ color: #1f7a1f; }}
  .delta-neg {{ color: #b00; }}
  .pill {{ display: inline-block; padding: 1px 8px; border-radius: 10px; background: #eef; font-size: 12px; }}
  .footnote {{ color: #777; font-size: 12px; margin-top: 12px; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>GraphRAG-Bench — {adapter_name}</h1>
<div class="meta">
  <span><strong>Run:</strong> <code>{run_id}</code></span>
  <span><strong>Sample:</strong> <span class="pill">{sample_mode}</span></span>
  <span><strong>Questions:</strong> {num_questions}</span>
  <span><strong>Documents:</strong> {num_documents}</span>
  <span><strong>Runtime:</strong> {runtime_min:.1f} min</span>
  <span><strong>Cost:</strong> ${cost_usd:.2f}</span>
  <span><strong>Judge:</strong> <code>{judge_model}</code></span>
</div>

{reliability_banner}
{construction_section}

<h2>Headline metrics — your run vs Khora reference baseline</h2>
<table>
  <thead><tr><th>metric</th><th class="num">your run</th><th class="num">Khora reference</th><th class="num">Δ</th></tr></thead>
  <tbody>{headline_rows}</tbody>
</table>
<div class="footnote">Reference baseline: khora <code>{ref_version}</code> on <code>{ref_dataset}</code> at <code>full</code> sampling with <code>{ref_judge}</code> judge.</div>

{difficulty_section}
{type_section}
{errors_section}
</body>
</html>
"""


def _fmt_num(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta_cell(local, ref):
    if local is None:
        return '<td class="num">—</td>'
    diff = local - ref
    pct = (diff / ref * 100) if ref else 0.0
    cls = "delta-pos" if diff >= 0 else "delta-neg"
    sign = "+" if diff >= 0 else ""
    return f'<td class="num {cls}">{sign}{diff:.3f} ({sign}{pct:.1f}%)</td>'


def write_html_report(result: BenchmarkRunResult, out_dir: str | Path) -> Path:
    """Render a single-file HTML report to ``{out_dir}/report.html``."""
    out_path = Path(out_dir) / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agg = result.aggregate_metrics

    # Headline rows
    headline_keys = [
        ("mean_answer_score", "mean_answer_score"),
        ("accuracy", "accuracy"),
        ("coverage", "coverage"),
        ("rouge_l", "rouge_l"),
        ("faithfulness", "faithfulness"),
        ("context_relevance", "context_relevance"),
        ("evidence_recall", "evidence_recall"),
    ]
    rows = []
    for label, key in headline_keys:
        ref = KHORA_BASELINE.get(key)
        if ref is None:
            continue
        local = agg.get(key)
        rows.append(
            f"<tr><td><code>{label}</code></td>"
            f'<td class="num">{_fmt_num(local)}</td>'
            f'<td class="num">{_fmt_num(ref)}</td>'
            f"{_delta_cell(local, ref)}</tr>"
        )
    if "cost_usd" in agg:
        ref = KHORA_BASELINE["cost_usd"]
        rows.append(
            f'<tr><td><code>cost_usd</code></td><td class="num">${agg["cost_usd"]:.2f}</td>'
            f'<td class="num">${ref:.2f}</td>{_delta_cell(agg["cost_usd"], ref)}</tr>'
        )
    ref_rt = float(KHORA_BASELINE["runtime_minutes"])
    rows.append(
        f'<tr><td><code>runtime_min</code></td><td class="num">{result.runtime_seconds / 60:.1f}</td>'
        f'<td class="num">{ref_rt:.0f}</td>{_delta_cell(result.runtime_seconds / 60, ref_rt)}</tr>'
    )

    # Construction section
    construction_html = ""
    if result.construction is not None:
        c = result.construction
        construction_html = (
            "<h2>Phase 1 — graph construction</h2>"
            "<table><tbody>"
            f"<tr><td>nodes</td><td class='num'>{c.num_nodes:,}</td></tr>"
            f"<tr><td>edges</td><td class='num'>{c.num_edges:,}</td></tr>"
            f"<tr><td>communities</td><td class='num'>{c.num_communities}</td></tr>"
            f"<tr><td>avg degree</td><td class='num'>{c.avg_degree:.2f}</td></tr>"
            f"<tr><td>construction time</td><td class='num'>{c.construction_time_ms / 1000:.1f} s</td></tr>"
            "</tbody></table>"
        )

    # Difficulty / type breakdowns
    def _breakdown_html(title: str, table: dict[str, dict[str, float]]) -> str:
        if not table:
            return ""
        body_rows = []
        for label, m in sorted(table.items()):
            body_rows.append(
                f"<tr><td><code>{label}</code></td>"
                f"<td class='num'>{int(m.get('n', 0))}</td>"
                f"<td class='num'>{_fmt_num(m.get('accuracy', 0))}</td></tr>"
            )
        return (
            f"<h2>{title}</h2><table>"
            "<thead><tr><th>label</th><th class='num'>n</th>"
            "<th class='num'>accuracy</th></tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    difficulty_html = _breakdown_html("Breakdown by difficulty", result.by_difficulty)
    type_html = _breakdown_html("Breakdown by question type", result.by_question_type)

    errors_html = ""
    if result.errors:
        items = "".join(f"<li><code>{err}</code></li>" for err in result.errors[:20])
        more = f"<li>… and {len(result.errors) - 20} more</li>" if len(result.errors) > 20 else ""
        errors_html = f"<h2>Errors</h2><ul>{items}{more}</ul>"

    reliability_html = ""
    error_rate = agg.get("error_rate", 0.0)
    if error_rate > 0:
        error_count = int(agg.get("error_count", 0))
        reliability_html = (
            '<div class="footnote" style="border-left:4px solid #c33;padding-left:10px">'
            f"⚠️ <strong>{error_count}/{result.num_questions} questions errored "
            f"({error_rate * 100:.1f}%)</strong> and were excluded from the metrics below — "
            "aggregates cover the successful questions only and are "
            "<strong>not comparable to the reference baseline</strong> if this rate is non-trivial."
            "</div>"
        )

    html = _TEMPLATE.format(
        reliability_banner=reliability_html,
        adapter_name=result.adapter_name,
        run_id=result.run_id,
        sample_mode=result.sample_mode,
        num_questions=result.num_questions,
        num_documents=result.num_documents,
        runtime_min=result.runtime_seconds / 60,
        cost_usd=result.cost_usd,
        judge_model=result.judge_model,
        construction_section=construction_html,
        headline_rows="".join(rows),
        ref_version=KHORA_BASELINE["khora_version"],
        ref_dataset=KHORA_BASELINE["dataset"],
        ref_judge=KHORA_BASELINE["judge_model"],
        difficulty_section=difficulty_html,
        type_section=type_html,
        errors_section=errors_html,
    )
    out_path.write_text(html)
    return out_path
