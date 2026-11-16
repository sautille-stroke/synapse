# NOT FOR PRODUCTION - LOCAL DEV PREVIEW ONLY
# This standalone developer preview script is not part of the synapse runtime SDK
# and must never be run in production environments (Zero-Python server-side rule).
"""Local developer preview web server for Synapse Dataset Viewer."""

from __future__ import annotations

import argparse
import http.server
import signal
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.viewer.static_html import VIEWER_HTML_TEMPLATE

ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALLOWED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".html", ".json", ".png", ".svg", ".jpg", ".jpeg", ".mp4"}
)


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

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        self.dataset_dir = Path(directory).resolve() if directory else Path.cwd().resolve()
        super().__init__(*args, directory=str(self.dataset_dir), **kwargs)

    def do_GET(self) -> None:
        # Decode percent-encoded characters before path traversal checking
        raw_path = self.path.split("?")[0].strip()
        decoded_path = urllib.parse.unquote(raw_path)
        clean_path = decoded_path.lstrip("/").replace("\\", "/")

        if clean_path in ("", "index.html"):
            content = VIEWER_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
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

        # Serve static file with security headers
        return super().do_GET()

    def end_headers(self) -> None:
        # Add security headers
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.path.lower().endswith(".svg"):
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'",
            )
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


class ViewerServer:
    """Local HTTP server serving the Synapse dataset viewer (Dev preview only)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        port: int = 0,
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

        # Port binding without probing blind range:
        # When port=0, OS assigns an ephemeral free port.
        # When port>0, bind directly; if in use, fail closed with OSError.
        self.httpd = _ReusableHTTPServer(
            (self.host, self.port), handler_cls,  # type: ignore[arg-type]
        )
        self.port = self.httpd.server_address[1]

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

        old_sigint: Any = None
        old_sigterm: Any = None
        try:
            old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _sig_handler)
            if hasattr(signal, "SIGTERM"):
                old_sigterm = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, _sig_handler)
        except (ValueError, AttributeError):
            pass

        try:
            self._stop_event.wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            try:
                if old_sigint is not None:
                    signal.signal(signal.SIGINT, old_sigint)
                if old_sigterm is not None and hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, old_sigterm)
            except (ValueError, AttributeError):
                pass
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
    port: int = 0,
    open_browser: bool = True,
    block: bool = True,
    host: str = "127.0.0.1",
) -> ViewerServer:
    """One-click entry point to launch the preview viewer."""
    server = ViewerServer(dataset_dir, port=port, host=host)
    url = server.start(open_browser=open_browser)
    print(f"\n[Synapse Viewer] Running at: {url}")
    print(f"[Synapse Viewer] Serving dataset from: {Path(dataset_dir).resolve()}\n")

    if block:
        print("Press Ctrl+C to stop.")
        server.serve_forever()

    return server


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dev_preview_viewer",
        description="Local preview viewer server for Synapse datasets (DEV ONLY).",
    )
    parser.add_argument("dataset", help="Path to dataset directory")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    args = parser.parse_args()

    launch_viewer(
        args.dataset,
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        block=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
