"""Self-contained HTML report generator for 3D model evaluation.

Produces a single HTML file with zero external dependencies: all CSS
is inlined, all charts are inline SVG, screenshots are base64-encoded
PNG ``<img>`` tags.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path

# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------


@dataclass
class TopologyResult:
    """Topology analysis results for a single mesh."""

    is_manifold: bool
    is_watertight: bool
    num_vertices: int
    num_faces: int
    num_connected_components: int
    has_degenerate_faces: bool
    num_boundary_edges: int


@dataclass
class ModelEvalResult:
    """Evaluation results for a single 3D model."""

    model_name: str
    model_path: str
    topology: TopologyResult
    metrics: dict[str, float]
    gate_results: dict[str, bool]
    screenshots_base64: list[str]
    screenshot_labels: list[str]
    overall_passed: bool
    error: str | None = None


@dataclass
class EvalReportData:
    """Top-level evaluation report data."""

    timestamp: str
    profile_name: str
    models: list[ModelEvalResult] = field(default_factory=list)
    reference_dir: str | None = None


# -------------------------------------------------------------------
# Metric normalisation (for radar chart display)
# -------------------------------------------------------------------


# Raw scale/count fields displayed in tables, NOT plotted as [0, 1] quality axes
_NON_QUALITY_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "vertex_count",
        "face_count",
        "num_vertices",
        "num_faces",
    }
)


def _normalize_metric(name: str, value: float) -> float:
    """Normalise a metric value to ``[0, 1]`` quality score (1.0 = best).

    Parameters
    ----------
    name:
        Metric name (e.g. ``chamfer_mm``, ``silhouette_iou``).
    value:
        Raw metric value.

    Returns
    -------
    float
        Normalised quality score in ``[0, 1]`` where ``1.0`` is optimal.
    """
    if not math.isfinite(value):
        return 0.0
    if name == "chamfer_mm":
        return max(0.0, 1.0 - min(max(value, 0.0) / 10.0, 1.0))
    if name == "volume_breathing":
        return max(0.0, 1.0 - min(abs(value) / 0.5, 1.0))
    if name == "volume_breathing_pct":
        return max(0.0, 1.0 - min(abs(value) / 10.0, 1.0))
    if name in ("silhouette_iou", "f_score", "f_score_1mm", "ssim"):
        return max(0.0, min(value, 1.0))
    if name == "psnr_db":
        return max(0.0, min(value / 50.0, 1.0))
    if name == "lpips":
        return max(0.0, 1.0 - min(max(value, 0.0), 1.0))
    if name in (
        "ciede2000",
        "albedo_delta_e00",
        "border_ring_ciede2000",
        "style_color_de",
    ):
        return max(0.0, 1.0 - min(max(value, 0.0) / 10.0, 1.0))
    if name in ("is_manifold", "is_watertight"):
        return 1.0 if value >= 0.5 else 0.0
    if name in (
        "num_degenerate_faces",
        "degenerate_faces",
        "boundary_edges",
        "num_boundary_edges",
        "self_intersecting_pairs",
        "floating_island_faces",
    ):
        # 0 defects = 1.0; monotonically decays toward 0 as defects grow
        return 1.0 / (1.0 + max(0.0, value))
    if name in ("connected_components", "num_connected_components"):
        if value <= 0.0:
            return 0.0
        return min(1.0, 1.0 / value)
    # Default: assume higher is better in [0, 1]
    return max(0.0, min(value, 1.0))


# -------------------------------------------------------------------
# SVG chart helpers
# -------------------------------------------------------------------


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


def _check_icon(ok: bool) -> str:
    """Return green checkmark or red cross."""
    if ok:
        return '<span style="color:#22c55e;font-size:16px;">&#10003;</span>'
    return '<span style="color:#ef4444;font-size:16px;">&#10007;</span>'


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
            f'font-size="11" fill="#333" '
            f'dominant-baseline="central">'
            f'{label_text}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


# -------------------------------------------------------------------
# HTML report generation
# -------------------------------------------------------------------


def generate_eval_html_report(
    data: EvalReportData,
    output_path: Path,
) -> Path:
    """Generate a self-contained HTML 3D model evaluation report.

    Parameters
    ----------
    data:
        Evaluation results for all models.
    output_path:
        Destination path for the HTML file.

    Returns
    -------
    Path
        The written output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(data.models)
    n_pass = sum(1 for m in data.models if m.overall_passed)
    n_fail = n_total - n_pass

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append(
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">'
    )
    parts.append("<title>Synapse 3D Model Evaluation Report</title>")
    parts.append("<style>")
    parts.append(_CSS)
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")

    # ---- Header ----
    parts.append("<header>")
    parts.append("<h1>Synapse 3D Model Evaluation Report</h1>")
    parts.append(
        f'<p class="subtitle">{html.escape(data.timestamp)} | '
        f'Profile: {html.escape(data.profile_name)} | '
        f'{n_total} models | '
        f'<span class="pass-count">{n_pass} passed</span> / '
        f'<span class="fail-count">{n_fail} failed</span></p>'
    )
    parts.append("</header>")
    parts.append("<main>")

    # ---- Summary table ----
    parts.append('<section class="summary">')
    parts.append("<h2>Summary</h2>")
    parts.append('<table class="summary-table">')
    parts.append(
        "<thead><tr>"
        "<th>Model</th><th>Status</th>"
        "<th>Vertices</th><th>Faces</th>"
        "<th>Manifold</th><th>Watertight</th>"
        "</tr></thead>"
    )
    parts.append("<tbody>")
    for m in data.models:
        name_esc = html.escape(m.model_name)
        parts.append(
            f"<tr><td>{name_esc}</td>"
            f"<td>{_status_badge(m.overall_passed)}</td>"
            f"<td>{m.topology.num_vertices:,}</td>"
            f"<td>{m.topology.num_faces:,}</td>"
            f"<td>{_check_icon(m.topology.is_manifold)}</td>"
            f"<td>{_check_icon(m.topology.is_watertight)}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("</section>")

    # ---- Aggregate radar chart ----
    if data.models:
        agg_scores = _compute_aggregate_scores(data.models)
        parts.append('<section class="radar">')
        parts.append("<h2>Aggregate Quality Scores</h2>")
        parts.append('<div class="chart-container">')
        parts.append(_svg_radar_chart(agg_scores, size=320))
        parts.append("</div>")
        parts.append("</section>")

    # ---- Per-model detail cards ----
    parts.append('<section class="details">')
    parts.append("<h2>Model Details</h2>")
    for m in data.models:
        _append_model_card(parts, m)
    parts.append("</section>")

    # ---- Footer ----
    parts.append(
        f'<footer><p>Generated by Synapse 3D Evaluation | '
        f'{html.escape(data.profile_name)}</p></footer>'
    )

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")

    html_content = "\n".join(parts)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def _append_model_card(parts: list[str], m: ModelEvalResult) -> None:
    """Append HTML for a single model detail card."""
    name_esc = html.escape(m.model_name)
    css_class = "card-pass" if m.overall_passed else "card-fail"
    parts.append(f'<div class="card {css_class}">')
    parts.append(f"<h3>{name_esc} {_status_badge(m.overall_passed)}</h3>")

    if m.error:
        parts.append(
            f'<p class="error">Error: {html.escape(m.error)}</p>'
        )

    # Screenshot gallery
    if m.screenshots_base64:
        parts.append('<div class="gallery">')
        for img_b64, label in zip(
            m.screenshots_base64, m.screenshot_labels
        ):
            parts.append('<div class="gallery-item">')
            parts.append(
                f'<img src="data:image/png;base64,{img_b64}" '
                f'alt="{html.escape(label)}" />'
            )
            parts.append(
                f'<span class="gallery-label">'
                f'{html.escape(label)}</span>'
            )
            parts.append("</div>")
        parts.append("</div>")

    # Topology summary
    t = m.topology
    parts.append('<table class="topo-table">')
    parts.append(
        "<thead><tr><th>Property</th><th>Value</th></tr></thead>"
    )
    parts.append("<tbody>")
    parts.append(
        f"<tr><td>Manifold</td>"
        f"<td>{_check_icon(t.is_manifold)}</td></tr>"
    )
    parts.append(
        f"<tr><td>Watertight</td>"
        f"<td>{_check_icon(t.is_watertight)}</td></tr>"
    )
    parts.append(
        f"<tr><td>Vertices</td><td>{t.num_vertices:,}</td></tr>"
    )
    parts.append(f"<tr><td>Faces</td><td>{t.num_faces:,}</td></tr>")
    parts.append(
        f"<tr><td>Connected Components</td>"
        f"<td>{t.num_connected_components}</td></tr>"
    )
    parts.append(
        f"<tr><td>Degenerate Faces</td>"
        f"<td>{_check_icon(not t.has_degenerate_faces)}</td></tr>"
    )
    parts.append(
        f"<tr><td>Boundary Edges</td>"
        f"<td>{t.num_boundary_edges}</td></tr>"
    )
    parts.append("</tbody></table>")

    # Metrics table with gate results
    if m.metrics:
        parts.append('<table class="metrics-table">')
        parts.append(
            "<thead><tr>"
            "<th>Metric</th><th>Value</th><th>Status</th>"
            "</tr></thead>"
        )
        parts.append("<tbody>")
        for mk, mv in sorted(m.metrics.items()):
            gate_passed = m.gate_results.get(mk)
            if gate_passed is not None:
                status_html = _status_badge(gate_passed)
            else:
                status_html = '<span style="color:#94a3b8;">—</span>'
            if isinstance(mv, float):
                val_str = f"{mv:.4f}"
            else:
                val_str = str(mv)
            parts.append(
                f"<tr><td>{html.escape(mk)}</td>"
                f"<td>{val_str}</td>"
                f"<td>{status_html}</td></tr>"
            )
        parts.append("</tbody></table>")

    # Per-model radar chart (exclude raw scale counts like vertex_count)
    if m.metrics:
        norm_scores = {
            k: _normalize_metric(k, v)
            for k, v in m.metrics.items()
            if isinstance(v, (int, float)) and k not in _NON_QUALITY_METRIC_KEYS
        }
        if norm_scores:
            parts.append('<div class="chart-container">')
            parts.append(_svg_radar_chart(norm_scores, size=280))
            parts.append("</div>")

    parts.append("</div>")


def _compute_aggregate_scores(
    models: list[ModelEvalResult],
) -> dict[str, float]:
    """Compute aggregate normalised quality scores across all models."""
    accum: dict[str, list[float]] = {}
    for m in models:
        # Topology scores
        accum.setdefault("manifold", []).append(
            1.0 if m.topology.is_manifold else 0.0
        )
        accum.setdefault("watertight", []).append(
            1.0 if m.topology.is_watertight else 0.0
        )
        accum.setdefault("clean_topo", []).append(
            1.0 if not m.topology.has_degenerate_faces else 0.0
        )
        # Metric-derived quality scores (skip raw counts & duplicate topo flags)
        for k, v in m.metrics.items():
            if (
                isinstance(v, (int, float))
                and k not in _NON_QUALITY_METRIC_KEYS
                and k not in ("is_manifold", "is_watertight")
            ):
                accum.setdefault(k, []).append(
                    _normalize_metric(k, v)
                )

    return {
        k: sum(vals) / len(vals) if vals else 0.0
        for k, vals in accum.items()
    }


# -------------------------------------------------------------------
# Inlined CSS
# -------------------------------------------------------------------

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Roboto, sans-serif;
  background: #f8fafc; color: #1e293b; line-height: 1.6;
  max-width: 1100px; margin: 0 auto; padding: 24px;
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
.topo-table { max-width: 400px; margin: 12px 0; }
.metrics-table { max-width: 500px; margin: 12px 0; }
.gallery {
  display: flex; flex-wrap: wrap; gap: 12px;
  margin: 12px 0; padding: 8px 0;
}
.gallery-item {
  display: flex; flex-direction: column; align-items: center;
}
.gallery-item img {
  width: 200px; height: 200px; object-fit: contain;
  border: 1px solid #e2e8f0; border-radius: 4px;
  background: #1e293b;
}
.gallery-label {
  font-size: 11px; color: #64748b; margin-top: 4px;
}
footer {
  margin-top: 32px; padding-top: 16px;
  border-top: 1px solid #e2e8f0;
  text-align: center; color: #94a3b8; font-size: 12px;
}
section { margin-bottom: 24px; }
"""
