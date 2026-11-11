"""synapse.viewer — Interactive web viewer and dashboard for multi-channel video datasets."""

from synapse.viewer.server import ViewerServer, launch_viewer

__all__ = [
    "ViewerServer",
    "launch_viewer",
]
