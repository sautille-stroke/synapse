# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""HTTP security utilities: URL validation, SSRF prevention, and safe openers."""

from __future__ import annotations

import ipaddress
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from synapse.errors import ContractValidationError

# Link-local / metadata networks to block (SSRF protection)
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
)
_BLOCKED_HOSTNAMES = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "instance-data",
    }
)


def validate_safe_url(url: str) -> str:
    """Validate that a URL uses safe HTTP/HTTPS scheme and does not target
    link-local or cloud metadata endpoints (SSRF prevention).

    Raises:
        ContractValidationError: If scheme is not http/https or target is blocked.
    """
    if not url or not isinstance(url, str):
        raise ContractValidationError(
            f"Invalid or empty URL: {url!r}",
            agent_fix_hint="Provide a non-empty HTTP/HTTPS URL.",
        )
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ContractValidationError(
            f"Blocked unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)",
            agent_fix_hint="Ensure the URL uses 'http://' or 'https://'.",
        )
    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise ContractValidationError(
            f"URL has no valid host: {url}",
            agent_fix_hint="Check the URL format.",
        )
    if hostname in _BLOCKED_HOSTNAMES:
        raise ContractValidationError(
            f"Blocked link-local / cloud metadata host: {hostname}",
            agent_fix_hint="Access to cloud metadata endpoints (169.254.x.x) is forbidden.",
        )
    try:
        ip = ipaddress.ip_address(hostname)
        for net in _BLOCKED_NETWORKS:
            if ip in net:
                raise ContractValidationError(
                    f"Blocked link-local / cloud metadata IP: {ip}",
                    agent_fix_hint="Access to cloud metadata endpoints (169.254.x.x) is forbidden.",
                )
    except ValueError:
        pass  # Hostname is not an IP literal
    return url


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP redirect handler that validates target URLs against SSRF."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener with SafeRedirectHandler installed."""
    return urllib.request.build_opener(SafeRedirectHandler())


def safe_urlopen(
    url_or_req: str | urllib.request.Request,
    timeout: float = 30.0,
) -> Any:
    """Open a URL or Request using safe redirect handling and URL validation."""
    if isinstance(url_or_req, str):
        validate_safe_url(url_or_req)
        req = urllib.request.Request(url_or_req)
    else:
        validate_safe_url(url_or_req.full_url)
        req = url_or_req

    opener = build_safe_opener()
    return opener.open(req, timeout=timeout)
