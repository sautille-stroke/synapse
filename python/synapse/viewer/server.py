"""Lightweight local web server and static HTML exporter for Synapse Dataset Viewer."""

from __future__ import annotations

import base64
import http.server
import json
import signal
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any

from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.viewer.static_html import VIEWER_HTML_TEMPLATE

ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_MEDIA_EXTENSIONS: frozenset[str] = frozenset({".html", ".json", ".png", ".svg"})


class _ReusableHTTPServer(socketserver.TCPServer):
    """Local TCP server with allow_reuse_address set on instance, not globally."""

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
        bind_and_activate: bool = True,
    ) -> None:
        self.allow_reuse_address = True
        super().__init__(
            server_address,
            RequestHandlerClass,
            bind_and_activate=bind_and_activate,
        )


class _DatasetHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the viewer UI and dataset media files."""

    def __init__(self, *args, directory: str | None = None, **kwargs) -> None:
        self.dataset_dir = Path(directory).resolve() if directory else Path.cwd().resolve()
        super().__init__(*args, directory=str(self.dataset_dir), **kwargs)

    def do_GET(self) -> None:
        # Clean path: strip query params, normalize slashes
        raw_path = self.path.split("?")[0].strip()
        clean_path = raw_path.lstrip("/")

        if clean_path in ("", "index.html"):
            # Serve the interactive single-page viewer
            content = VIEWER_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        # Reject path traversal (e.g. '..') and hidden files (starting with '.')
        path_obj = Path(clean_path)
        for part in path_obj.parts:
            if part == ".." or part.startswith("."):
                self.send_error(403, "Access denied: Path traversal or hidden file")
                return

        try:
            target_file = (self.dataset_dir / path_obj).resolve()
            target_file.relative_to(self.dataset_dir)
        except (ValueError, RuntimeError):
            self.send_error(403, "Access denied: Path outside dataset directory")
            return

        if target_file.is_dir():
            self.send_error(403, "Access denied: Directory listing forbidden")
            return

        if target_file.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
            self.send_error(
                403,
                f"Access denied: File extension '{target_file.suffix}' not allowed",
            )
            return

        if not target_file.is_file():
            self.send_error(404, "File not found")
            return

        # Serve allowed static file from dataset directory
        return super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        # Mute standard noisy HTTP access logs, only log on errors if needed
        pass


class ViewerServer:
    """Local HTTP server serving the Synapse dataset viewer."""

    def __init__(
        self,
        dataset_dir: str | Path,
        port: int = 8765,
        host: str = "127.0.0.1",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.port = port
        self.host = host
        self.httpd: _ReusableHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if self.host not in ALLOWED_HOSTS:
            raise ContractValidationError(
                f"Non-loopback host '{self.host}' is rejected for security. "
                f"Host must be in {sorted(ALLOWED_HOSTS)}.",
                agent_fix_hint="Use loopback host: 127.0.0.1 or localhost.",
                details={"host": self.host, "allowed_hosts": sorted(ALLOWED_HOSTS)},
            )

    def start(self, open_browser: bool = False) -> str:
        """Start server in background thread and return access URL."""
        if not self.dataset_dir.exists():
            raise AssetNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}",
                agent_fix_hint=f"Ensure '{self.dataset_dir}' exists and contains manifest.json.",
            )

        def handler_cls(*args: Any, **kwargs: Any) -> _DatasetHTTPRequestHandler:
            return _DatasetHTTPRequestHandler(
                *args, directory=str(self.dataset_dir), **kwargs
            )

        # Try requested port or fallback to next available
        cur_port = self.port
        for _attempt in range(10):
            try:
                self.httpd = _ReusableHTTPServer(
                    (self.host, cur_port), handler_cls,  # type: ignore[arg-type]
                )
                self.port = cur_port
                break
            except OSError:
                cur_port += 1

        if self.httpd is None:
            raise RuntimeError(f"Could not bind to port {self.port} or subsequent 10 ports.")

        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://{self.host}:{self.port}"
        if open_browser:
            webbrowser.open(url)

        return url

    def serve_forever(self) -> None:
        """Block until server is stopped or interrupt signal received."""
        if self.httpd is None:
            self.start(open_browser=False)

        def _sig_handler(signum: int, frame: object) -> None:
            self.stop()

        try:
            signal.signal(signal.SIGINT, _sig_handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _sig_handler)
        except (ValueError, AttributeError):
            pass

        try:
            self._stop_event.wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Shut down the server."""
        self._stop_event.set()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def launch_viewer(
    dataset_dir: str | Path,
    port: int = 8765,
    open_browser: bool = True,
    block: bool = True,
    host: str = "127.0.0.1",
) -> ViewerServer:
    """One-click entry point to launch the viewer."""
    server = ViewerServer(dataset_dir, port=port, host=host)
    url = server.start(open_browser=open_browser)
    print(f"\n[Synapse Viewer] Running at: {url}")
    print(f"[Synapse Viewer] Serving dataset from: {Path(dataset_dir).resolve()}\n")

    if block:
        print("Press Ctrl+C to stop.")
        server.serve_forever()

    return server


def _encode_data_uri(file_path: Path) -> str | None:
    """Encode an image file to a base64 Data URI string."""
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".png":
            mime = "image/png"
        elif suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".svg":
            mime = "image/svg+xml"
        else:
            return None
        raw_bytes = file_path.read_bytes()
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def export_static_html(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained static HTML viewer file with embedded dataset data.

    Embeds manifest.json and base64-encoded images into the HTML template,
    requiring zero running processes or local web servers to view.

    Args:
        dataset_dir: Directory containing manifest.json and frame/visualization media.
        output_path: Destination path for the .html file. If None, defaults to
            ``dataset_dir / "viewer.html"``.

    Returns:
        Path to the generated self-contained HTML file.
    """
    dataset_path = Path(dataset_dir).resolve()
    if not dataset_path.exists():
        raise AssetNotFoundError(
            f"Dataset directory not found: {dataset_path}",
            agent_fix_hint=f"Ensure '{dataset_path}' exists and contains manifest.json.",
        )

    manifest_file = dataset_path / "manifest.json"
    if not manifest_file.is_file():
        raise AssetNotFoundError(
            f"manifest.json not found in dataset directory: {dataset_path}",
            agent_fix_hint="Generate a dataset first using 'synapse generate'.",
        )

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Gather images to embed as base64
    embedded_images: dict[str, str] = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".svg"}
    for subdir_name in ["frames", "visualizations"]:
        subdir = dataset_path / subdir_name
        if subdir.is_dir():
            for img_file in sorted(subdir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    rel_key = f"{subdir_name}/{img_file.name}"
                    uri = _encode_data_uri(img_file)
                    if uri:
                        embedded_images[rel_key] = uri

    contact_sheet = dataset_path / "contact_sheet.png"
    if contact_sheet.is_file():
        uri = _encode_data_uri(contact_sheet)
        if uri:
            embedded_images["contact_sheet.png"] = uri

    html_content = VIEWER_HTML_TEMPLATE
    html_content = html_content.replace(
        "/*__EMBEDDED_MANIFEST__*/ null",
        json.dumps(manifest_data),
    )
    html_content = html_content.replace(
        "/*__EMBEDDED_IMAGES__*/ {}",
        json.dumps(embedded_images),
    )

    if output_path is None:
        target_path = dataset_path / "viewer.html"
    else:
        target_path = Path(output_path).resolve()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(html_content, encoding="utf-8")
    return target_path
