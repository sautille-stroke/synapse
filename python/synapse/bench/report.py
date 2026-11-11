"""Self-contained HTML report generator for Synapse-Bench.

Produces a single HTML file with zero external dependencies: all CSS
is inlined, all charts are inline SVG, no ``http://`` or ``https://``
references.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        f'border-radius:4px;color:#fff;font-weight:700;'
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
    lines.append(
        f'<svg viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">'
    )

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
            f'  <polygon points="{" ".join(pts)}" '
            f'fill="none" stroke="#ddd" stroke-width="1"/>'
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
        lines.append(
            f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="3" '
            f'fill="#3b82f6"/>'
        )
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
            f'{label_text}</text>'
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
    lines.append(
        f'<svg viewBox="0 0 {width} {actual_h}" '
        f'width="{width}" height="{actual_h}">'
    )

    if title:
        lines.append(
            f'  <text x="{width // 2}" y="18" text-anchor="middle" '
            f'font-size="14" font-weight="700" fill="#333">'
            f'{html.escape(title)}</text>'
        )

    for i, (label, val) in enumerate(zip(labels, values)):
        y = margin_top + i * (bar_height + bar_gap)
        bw = max(1, int(bar_area_w * val / max_val))
        # Label
        lines.append(
            f'  <text x="{margin_left - 6}" y="{y + bar_height // 2 + 1}" '
            f'text-anchor="end" font-size="11" fill="#555" '
            f'dominant-baseline="central">'
            f'{html.escape(label[:20])}</text>'
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
            f'{val:.1f}</text>'
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

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    n_total = len(reports)
    n_pass = sum(
        1 for r in reports if r.error is None
        and (
            r.gate_verdict is None or r.gate_verdict.passed
        )
    )
    n_fail = n_total - n_pass

    # Compute pillar scores for radar chart
    pillar_scores = _compute_pillar_scores(reports)

    # Build HTML
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append("<meta charset=\"utf-8\">")
    parts.append(
        "<meta name=\"viewport\" "
        "content=\"width=device-width, initial-scale=1\">"
    )
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
        f"<p class=\"subtitle\">{timestamp} | "
        f"{n_total} scenarios | "
        f"<span class=\"pass-count\">{n_pass} passed</span> / "
        f"<span class=\"fail-count\">{n_fail} failed</span></p>"
    )
    parts.append("</header>")
    parts.append("<main>")

    # Summary table
    parts.append("<section class=\"summary\">")
    parts.append("<h2>Summary</h2>")
    parts.append("<table class=\"summary-table\">")
    parts.append(
        "<thead><tr>"
        "<th>Scenario</th><th>Status</th>"
        "<th>Adapter</th><th>Frames</th>"
        "<th>Total Time (ms)</th>"
        "</tr></thead>"
    )
    parts.append("<tbody>")
    for r in reports:
        passed = r.error is None and (
            r.gate_verdict is None or r.gate_verdict.passed
        )
        name_esc = html.escape(r.scenario_name)
        adapter_esc = html.escape(r.adapter_id)
        total_ms = r.timings.get("total_ms", 0.0)
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
    parts.append("<section class=\"radar\">")
    parts.append("<h2>Pillar Scores</h2>")
    parts.append(
        '<div class="chart-container">'
    )
    parts.append(_svg_radar_chart(pillar_scores, size=320))
    parts.append("</div>")
    parts.append("</section>")

    # Per-scenario detail cards
    parts.append("<section class=\"details\">")
    parts.append("<h2>Scenario Details</h2>")
    for r in reports:
        passed = r.error is None and (
            r.gate_verdict is None or r.gate_verdict.passed
        )
        name_esc = html.escape(r.scenario_name)
        parts.append(
            f'<div class="card {"card-pass" if passed else "card-fail"}">'
        )
        parts.append(
            f"<h3>{name_esc} {_status_badge(passed)}</h3>"
        )
        if r.error:
            parts.append(
                f'<p class="error">Error: '
                f'{html.escape(r.error)}</p>'
            )
        if r.metrics:
            parts.append("<table class=\"metrics-table\">")
            parts.append(
                "<thead><tr><th>Metric</th>"
                "<th>Value</th></tr></thead>"
            )
            parts.append("<tbody>")
            for mk, mv in sorted(r.metrics.items()):
                parts.append(
                    f"<tr><td>{html.escape(mk)}</td>"
                    f"<td>{mv:.4f}</td></tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</div>")
    parts.append("</section>")

    # Timing bar chart
    parts.append("<section class=\"timing\">")
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
    total_cpu_ms = sum(
        r.timings.get("total_ms", 0.0) for r in reports
    )
    adapters = sorted({r.adapter_id for r in reports})
    adapter_str = ", ".join(html.escape(a) for a in adapters)
    parts.append(
        f'<footer><p>Total CPU time: {total_cpu_ms:.0f} ms | '
        f'Adapters: {adapter_str}</p></footer>'
    )

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")

    html_content = "\n".join(parts)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def _compute_pillar_scores(
    reports: list[Any],
) -> dict[str, float]:
    """Derive 4-pillar aggregate scores from scenario reports.

    Pillars:
    - temporal: Based on flicker_score and temporal_warp_error
    - fgpp_fidelity: Based on depth_rmse_m, normal_angular_error_deg,
      silhouette_iou
    - roundtrip: Based on chamfer_mm, psnr_db, ssim
    - style: Based on style_color_de, style_light_error_deg
    """
    pillar_accumulators: dict[str, list[float]] = {
        "temporal": [],
        "fgpp_fidelity": [],
        "roundtrip": [],
        "style": [],
    }

    for r in reports:
        m = r.metrics
        if not m:
            continue

        # Temporal
        if "flicker_score" in m:
            # Lower is better; clamp to [0,1]
            pillar_accumulators["temporal"].append(
                max(0.0, 1.0 - min(m["flicker_score"], 1.0))
            )
        if "temporal_warp_error" in m:
            pillar_accumulators["temporal"].append(
                max(0.0, 1.0 - min(m["temporal_warp_error"], 1.0))
            )

        # FGPP fidelity
        if "silhouette_iou" in m:
            pillar_accumulators["fgpp_fidelity"].append(
                min(m["silhouette_iou"], 1.0)
            )
        if "depth_rmse_m" in m:
            pillar_accumulators["fgpp_fidelity"].append(
                max(0.0, 1.0 - min(m["depth_rmse_m"], 1.0))
            )

        # Roundtrip
        if "psnr_db" in m:
            # Normalize: 40 dB = 1.0, 0 dB = 0.0
            pillar_accumulators["roundtrip"].append(
                min(1.0, max(0.0, m["psnr_db"] / 40.0))
            )
        if "ssim" in m:
            pillar_accumulators["roundtrip"].append(
                min(1.0, max(0.0, m["ssim"]))
            )

        # Style
        if "style_color_de" in m:
            # Lower is better; 0 = perfect
            pillar_accumulators["style"].append(
                max(0.0, 1.0 - min(m["style_color_de"] / 20.0, 1.0))
            )
        if "style_light_error_deg" in m:
            pillar_accumulators["style"].append(
                max(
                    0.0,
                    1.0 - min(
                        m["style_light_error_deg"] / 45.0, 1.0,
                    ),
                )
            )

    scores: dict[str, float] = {}
    for pillar, vals in pillar_accumulators.items():
        if vals:
            scores[pillar] = sum(vals) / len(vals)
        else:
            scores[pillar] = 0.0

    return scores


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
