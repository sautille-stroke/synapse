"""Self-contained HTML report generator for Synapse-Bench.

Produces a single HTML file with zero external dependencies: all CSS
is inlined, all charts are inline SVG, no ``http://`` or ``https://``
references.
"""

from __future__ import annotations

import html
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synapse.observability import get_probe


def _status_badge(passed: bool) -> str:
    """Return an HTML span with a green/red pass/fail badge."""
    if passed:
        color = "#22c55e"
        label = "PASS"
    else:
        color = "#ef4444"
        label = "FAIL"
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f"border-radius:4px;color:#fff;font-weight:700;"
        f'font-size:13px;background:{color};">{label}</span>'
    )


def _svg_radar_chart(
    scores: dict[str, float],
    size: int = 300,
) -> str:
    """Generate an SVG radar chart for pillar scores.

    Parameters
    ----------
    scores:
        Mapping of pillar name to score (0.0 .. 1.0).
    size:
        SVG viewport dimension (square).

    Returns
    -------
    str
        Complete ``<svg>`` element string.
    """
    cx = size / 2
    cy = size / 2
    radius = size * 0.38
    labels = list(scores.keys())
    values = [max(0.0, min(1.0, scores.get(k, 0.0))) for k in labels]
    n = len(labels)

    if n < 3:
        # Need at least 3 axes for a radar chart
        labels = labels + [""] * (3 - n)
        values = values + [0.0] * (3 - n)
        n = 3

    angle_step = 2 * math.pi / n

    lines: list[str] = []
    lines.append(f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">')

    # Background grid rings
    for ring in [0.25, 0.5, 0.75, 1.0]:
        r = radius * ring
        pts = []
        for i in range(n):
            a = -math.pi / 2 + i * angle_step
            px = cx + r * math.cos(a)
            py = cy + r * math.sin(a)
            pts.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'  <polygon points="{" ".join(pts)}" fill="none" stroke="#ddd" stroke-width="1"/>'
        )

    # Axis lines
    for i in range(n):
        a = -math.pi / 2 + i * angle_step
        ex = cx + radius * math.cos(a)
        ey = cy + radius * math.sin(a)
        lines.append(
            f'  <line x1="{cx:.1f}" y1="{cy:.1f}" '
            f'x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="#ccc" stroke-width="1"/>'
        )

    # Data polygon
    pts = []
    for i in range(n):
        a = -math.pi / 2 + i * angle_step
        r = radius * values[i]
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        pts.append(f"{px:.1f},{py:.1f}")
    lines.append(
        f'  <polygon points="{" ".join(pts)}" '
        f'fill="rgba(59,130,246,0.25)" stroke="#3b82f6" '
        f'stroke-width="2"/>'
    )

    # Data dots and labels
    for i in range(n):
        a = -math.pi / 2 + i * angle_step
        r = radius * values[i]
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        lines.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="#3b82f6"/>')
        # Label position
        lx = cx + (radius + 20) * math.cos(a)
        ly = cy + (radius + 20) * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.3:
            anchor = "start"
        elif math.cos(a) < -0.3:
            anchor = "end"
        label_text = html.escape(labels[i])
        lines.append(
            f'  <text x="{lx:.1f}" y="{ly:.1f}" '
            f'text-anchor="{anchor}" '
            f'font-size="12" fill="#333" '
            f'dominant-baseline="central">'
            f"{label_text}</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _svg_bar_chart(
    data: dict[str, float],
    title: str = "",
    width: int = 600,
    height: int = 200,
) -> str:
    """Generate an SVG horizontal bar chart.

    Parameters
    ----------
    data:
        Mapping of bar label to numeric value.
    title:
        Chart title displayed above bars.
    width:
        SVG width in pixels.
    height:
        SVG height in pixels.

    Returns
    -------
    str
        Complete ``<svg>`` element string.
    """
    if not data:
        return ""

    labels = list(data.keys())
    values = [data[k] for k in labels]
    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0

    n = len(labels)
    margin_left = 140
    margin_top = 30 if title else 10
    margin_right = 60
    bar_height = 18
    bar_gap = 6
    chart_h = n * (bar_height + bar_gap) + margin_top + 10
    actual_h = max(height, chart_h)
    bar_area_w = width - margin_left - margin_right

    lines: list[str] = []
    lines.append(f'<svg viewBox="0 0 {width} {actual_h}" width="{width}" height="{actual_h}">')

    if title:
        lines.append(
            f'  <text x="{width // 2}" y="18" text-anchor="middle" '
            f'font-size="14" font-weight="700" fill="#333">'
            f"{html.escape(title)}</text>"
        )

    for i, (label, val) in enumerate(zip(labels, values, strict=False)):
        y = margin_top + i * (bar_height + bar_gap)
        bw = max(1, int(bar_area_w * val / max_val))
        # Label
        lines.append(
            f'  <text x="{margin_left - 6}" y="{y + bar_height // 2 + 1}" '
            f'text-anchor="end" font-size="11" fill="#555" '
            f'dominant-baseline="central">'
            f"{html.escape(label[:20])}</text>"
        )
        # Bar
        lines.append(
            f'  <rect x="{margin_left}" y="{y}" '
            f'width="{bw}" height="{bar_height}" '
            f'fill="#3b82f6" rx="3"/>'
        )
        # Value
        lines.append(
            f'  <text x="{margin_left + bw + 4}" '
            f'y="{y + bar_height // 2 + 1}" '
            f'font-size="11" fill="#333" '
            f'dominant-baseline="central">'
            f"{val:.1f}</text>"
        )

    lines.append("</svg>")
    return "\n".join(lines)


def generate_html_report(
    reports: list[Any],
    output_path: Path,
) -> Path:
    """Generate a self-contained HTML benchmark report.

    Parameters
    ----------
    reports:
        List of ``ScenarioRunReport`` instances.
    output_path:
        Destination path for the HTML file.

    Returns
    -------
    Path
        The written output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_total = len(reports)
    n_pass = sum(
        1 for r in reports if r.error is None and (r.gate_verdict is None or r.gate_verdict.passed)
    )
    n_fail = n_total - n_pass

    # Compute pillar scores for radar chart
    pillar_scores = compute_pillar_scores(reports)

    # Build HTML
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>Synapse-Bench Report</title>")
    parts.append("<style>")
    parts.append(_CSS)
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")

    # Header
    parts.append("<header>")
    parts.append("<h1>Synapse-Bench Report</h1>")
    parts.append(
        f'<p class="subtitle">{timestamp} | '
        f"{n_total} scenarios | "
        f'<span class="pass-count">{n_pass} passed</span> / '
        f'<span class="fail-count">{n_fail} failed</span></p>'
    )
    parts.append("</header>")
    parts.append("<main>")

    # Summary table
    parts.append('<section class="summary">')
    parts.append("<h2>Summary</h2>")
    parts.append('<table class="summary-table">')
    parts.append(
        "<thead><tr>"
        "<th>Scenario</th><th>Status</th>"
        "<th>Adapter</th><th>Frames</th>"
        "<th>Total Time (ms)</th>"
        "</tr></thead>"
    )
    parts.append("<tbody>")
    for r in reports:
        passed = r.error is None and (r.gate_verdict is None or r.gate_verdict.passed)
        name_esc = html.escape(r.scenario_name)
        adapter_esc = html.escape(r.adapter_id)
        total_ms_raw = r.timings.get("total_ms", 0.0)
        total_ms = 0.0 if (total_ms_raw is None or math.isnan(total_ms_raw)) else float(total_ms_raw)
        parts.append(
            f"<tr><td>{name_esc}</td>"
            f"<td>{_status_badge(passed)}</td>"
            f"<td>{adapter_esc}</td>"
            f"<td>{r.num_frames}</td>"
            f"<td>{total_ms:.1f}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("</section>")

    # Radar chart
    parts.append('<section class="radar">')
    parts.append("<h2>Pillar Scores</h2>")
    parts.append('<div class="chart-container">')
    parts.append(_svg_radar_chart(pillar_scores, size=320))
    parts.append("</div>")
    parts.append("</section>")

    # Per-scenario detail cards
    parts.append('<section class="details">')
    parts.append("<h2>Scenario Details</h2>")
    for r in reports:
        passed = r.error is None and (r.gate_verdict is None or r.gate_verdict.passed)
        name_esc = html.escape(r.scenario_name)
        parts.append(f'<div class="card {"card-pass" if passed else "card-fail"}">')
        parts.append(f"<h3>{name_esc} {_status_badge(passed)}</h3>")
        if r.error:
            parts.append(f'<p class="error">Error: {html.escape(r.error)}</p>')
        if r.metrics:
            parts.append('<table class="metrics-table">')
            parts.append("<thead><tr><th>Metric</th><th>Value</th></tr></thead>")
            parts.append("<tbody>")
            for mk, mv in sorted(r.metrics.items()):
                parts.append(f"<tr><td>{html.escape(mk)}</td><td>{mv:.4f}</td></tr>")
            parts.append("</tbody></table>")
        parts.append("</div>")
    parts.append("</section>")

    # Timing bar chart
    parts.append('<section class="timing">')
    parts.append("<h2>Timing Breakdown</h2>")
    for r in reports:
        if r.timings:
            name_esc = html.escape(r.scenario_name)
            parts.append(
                _svg_bar_chart(
                    r.timings,
                    title=name_esc,
                    width=600,
                    height=max(120, len(r.timings) * 28 + 40),
                )
            )
    parts.append("</section>")

    # Footer
    def _safe_t(val: Any) -> float:
        return 0.0 if (val is None or math.isnan(val)) else float(val)

    total_cpu_ms = sum(_safe_t(r.timings.get("total_ms", 0.0)) for r in reports)
    adapters = sorted({r.adapter_id for r in reports})
    adapter_str = ", ".join(html.escape(a) for a in adapters)
    parts.append(
        f"<footer><p>Total CPU time: {total_cpu_ms:.0f} ms | Adapters: {adapter_str}</p></footer>"
    )

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")

    html_content = "\n".join(parts)
    output_path.write_text(html_content, encoding="utf-8")

    # Log the HTML report as a Traceboard artifact
    html_bytes = html_content.encode("utf-8")
    probe = get_probe()
    if probe is not None and probe.active:
        probe.log_artifact(
            "bench/report",
            {
                "filename": output_path.name,
                "mime_type": "text/html",
                "size_bytes": len(html_bytes),
            },
            html_bytes,
        )

    return output_path


def compute_pillar_scores(
    reports: list[Any],
) -> dict[str, float]:
    """Derive 4-pillar aggregate scores from scenario reports (Single Source of Truth).

    Pillars:
    - temporal: Based on ``flicker_score`` and ``temporal_warp_error``
    - fgpp_fidelity: Based on ``depth_rmse_m``, ``normal_angular_error_deg``,
      ``silhouette_iou``
    - roundtrip: Based on ``chamfer_mm``, ``psnr_db``, ``ssim``
    - style: Based on ``style_color_de``, ``style_light_error_deg``
    """
    pillar_accumulators: dict[str, list[float]] = {
        "temporal": [],
        "fgpp_fidelity": [],
        "roundtrip": [],
        "style": [],
    }

    def _finite(val: Any) -> float | None:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None
        fval = float(val)
        return fval if math.isfinite(fval) else None

    for r in reports:
        m = getattr(r, "metrics", None)
        if not m:
            continue

        # Temporal
        if (v := _finite(m.get("flicker_score"))) is not None:
            pillar_accumulators["temporal"].append(max(0.0, 1.0 - min(v, 1.0)))
        if (v := _finite(m.get("temporal_warp_error"))) is not None:
            pillar_accumulators["temporal"].append(max(0.0, 1.0 - min(v, 1.0)))

        # FGPP fidelity
        if (v := _finite(m.get("silhouette_iou"))) is not None:
            pillar_accumulators["fgpp_fidelity"].append(max(0.0, min(v, 1.0)))
        if (v := _finite(m.get("depth_rmse_m"))) is not None:
            pillar_accumulators["fgpp_fidelity"].append(max(0.0, 1.0 - min(v, 1.0)))
        if (v := _finite(m.get("normal_angular_error_deg"))) is not None:
            pillar_accumulators["fgpp_fidelity"].append(max(0.0, 1.0 - min(v / 90.0, 1.0)))

        # Roundtrip
        if (v := _finite(m.get("psnr_db"))) is not None:
            pillar_accumulators["roundtrip"].append(min(1.0, max(0.0, v / 40.0)))
        if (v := _finite(m.get("ssim"))) is not None:
            pillar_accumulators["roundtrip"].append(min(1.0, max(0.0, v)))
        if (v := _finite(m.get("chamfer_mm"))) is not None:
            pillar_accumulators["roundtrip"].append(max(0.0, 1.0 - min(v / 50.0, 1.0)))

        # Style
        if (v := _finite(m.get("style_color_de"))) is not None:
            pillar_accumulators["style"].append(max(0.0, 1.0 - min(v / 20.0, 1.0)))
        if (v := _finite(m.get("style_light_error_deg"))) is not None:
            pillar_accumulators["style"].append(max(0.0, 1.0 - min(v / 45.0, 1.0)))

    return {
        pillar: (sum(vals) / len(vals) if vals else 0.0)
        for pillar, vals in pillar_accumulators.items()
    }


_compute_pillar_scores = compute_pillar_scores


# Inlined CSS for the report.
_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Roboto, sans-serif;
  background: #f8fafc; color: #1e293b; line-height: 1.6;
  max-width: 960px; margin: 0 auto; padding: 24px;
}
header { text-align: center; margin-bottom: 32px; }
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { font-size: 20px; margin: 24px 0 12px; }
h3 { font-size: 16px; margin-bottom: 8px; }
.subtitle { color: #64748b; font-size: 14px; }
.pass-count { color: #22c55e; font-weight: 700; }
.fail-count { color: #ef4444; font-weight: 700; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td {
  padding: 6px 12px; text-align: left; border-bottom: 1px solid #e2e8f0;
  font-size: 13px;
}
th { background: #f1f5f9; font-weight: 600; }
.summary-table { margin-bottom: 24px; }
.chart-container { text-align: center; margin: 16px 0; }
.card {
  border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px;
  margin: 12px 0; background: #fff;
}
.card-pass { border-left: 4px solid #22c55e; }
.card-fail { border-left: 4px solid #ef4444; }
.error { color: #ef4444; font-size: 13px; margin: 4px 0; }
.metrics-table { max-width: 400px; }
footer {
  margin-top: 32px; padding-top: 16px;
  border-top: 1px solid #e2e8f0;
  text-align: center; color: #94a3b8; font-size: 12px;
}
section { margin-bottom: 24px; }
"""


def render_comparison_html(
    comparison: Any,
    output_path: Path | None = None,
) -> str:
    """Render a self-contained HTML 4-track cross-model comparison leaderboard (PR-215)."""
    from synapse.bench.compare import TRACK_DEFINITIONS

    data = comparison.to_dict() if hasattr(comparison, "to_dict") else dict(comparison)
    rows = data.get("rows", [])
    track_winners = data.get("track_winners", {})
    overall_winner = str(data.get("overall_winner", ""))

    bar_data = {
        str(r.get("model_name", "")): round(float(r.get("overall_score", 0.0)) * 100.0, 2)
        for r in rows
    }
    svg_chart = (
        _svg_bar_chart(bar_data, title="Overall 4-Track Score (0-100)")
        if bar_data
        else ""
    )

    row_html: list[str] = []
    for rank, r in enumerate(rows, start=1):
        name = html.escape(str(r.get("model_name", "")))
        overall = float(r.get("overall_score", 0.0)) * 100.0
        ts = r.get("track_scores", {})
        t_cells = "".join(
            f"<td>{float(ts.get(tid, {}).get('score', 0.0)) * 100.0:.1f}</td>"
            for tid in TRACK_DEFINITIONS
        )
        row_html.append(f"<tr><td>#{rank}</td><td><strong>{name}</strong></td><td>{overall:.1f}</td>{t_cells}</tr>")

    winner_items = "".join(
        f"<li><strong>{html.escape(str(TRACK_DEFINITIONS[tid]['title']))}</strong>: "
        f"{html.escape(str(track_winners.get(tid, 'N/A')))}</li>"
        for tid in TRACK_DEFINITIONS
    )

    doc = "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Synapse-Bench 4-Track Leaderboard</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>Synapse-Bench 4-Track Comparison Leaderboard</h1>",
            f'<p class="subtitle">Overall Winner: <strong>{html.escape(overall_winner)}</strong></p>',
            "</header>",
            f'<div class="chart-container">{svg_chart}</div>',
            "<section>",
            "<h2>Leaderboard Ranking</h2>",
            '<table class="summary-table">',
            "<thead><tr><th>Rank</th><th>Model</th><th>Overall</th>"
            + "".join(f"<th>{html.escape(tid)}</th>" for tid in TRACK_DEFINITIONS)
            + "</tr></thead>",
            "<tbody>",
            *row_html,
            "</tbody>",
            "</table>",
            "</section>",
            "<section>",
            "<h2>Per-Track Winners</h2>",
            f"<ul>{winner_items}</ul>",
            "</section>",
            "</body>",
            "</html>",
        ]
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(doc, encoding="utf-8")
    return doc
