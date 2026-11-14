"""synapse.viewer — Interactive web viewer and dashboard for multi-channel video datasets."""

from synapse.viewer.server import ViewerServer, export_static_html, launch_viewer

__all__ = [
    "ViewerServer",
    "export_static_html",
    "launch_viewer",
]
