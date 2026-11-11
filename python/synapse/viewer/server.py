"""Lightweight local web server for Synapse Dataset Viewer."""

from __future__ import annotations

import http.server
import os
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from synapse.viewer.static_html import VIEWER_HTML_TEMPLATE


class _DatasetHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the viewer UI and dataset media files."""

    def __init__(self, *args, directory=None, **kwargs):
        self.dataset_dir = Path(directory).resolve() if directory else Path.cwd().resolve()
        super().__init__(*args, directory=str(self.dataset_dir), **kwargs)

    def do_GET(self):
        # Clean path
        path = self.path.split("?")[0].strip()

        if path in ("", "/", "/index.html"):
            # Serve the interactive single-page viewer
            content = VIEWER_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # Serve static media from dataset directory
        return super().do_GET()

    def log_message(self, format, *args):
        # Mute standard noisy HTTP access logs, only log on errors if needed
        pass


class ViewerServer:
    """Embedded HTTP server hosting the interactive dataset viewer."""

    def __init__(self, dataset_dir: str | Path, port: int = 8765):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.port = port
        self.httpd: Optional[socketserver.TCPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, open_browser: bool = False) -> str:
        """Start server in background thread and return access URL."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        handler_cls = lambda *args, **kwargs: _DatasetHTTPRequestHandler(
            *args, directory=str(self.dataset_dir), **kwargs
        )

        # Allow port reuse
        socketserver.TCPServer.allow_reuse_address = True
        
        # Try requested port or fallback to next available
        cur_port = self.port
        for attempt in range(10):
            try:
                self.httpd = socketserver.TCPServer(("", cur_port), handler_cls)
                self.port = cur_port
                break
            except OSError:
                cur_port += 1

        if self.httpd is None:
            raise RuntimeError(f"Could not bind to port {self.port} or subsequent 10 ports.")

        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://127.0.0.1:{self.port}"
        if open_browser:
            webbrowser.open(url)

        return url

    def stop(self):
        """Shut down the server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def launch_viewer(
    dataset_dir: str | Path,
    port: int = 8765,
    open_browser: bool = True,
    block: bool = True,
) -> ViewerServer:
    """One-click entry point to launch the viewer."""
    server = ViewerServer(dataset_dir, port=port)
    url = server.start(open_browser=open_browser)
    print(f"\n[Synapse Viewer] Running at: {url}")
    print(f"[Synapse Viewer] Serving dataset from: {Path(dataset_dir).resolve()}\n")

    if block:
        try:
            print("Press Ctrl+C to stop.")
            while True:
                threading.Event().wait(1.0)
        except KeyboardInterrupt:
            print("\nShutting down viewer...")
            server.stop()

    return server
