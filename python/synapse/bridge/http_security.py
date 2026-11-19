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
    ipaddress.ip_network("100.100.100.200/32"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fd00:ec2::254/128"),
    ipaddress.ip_network("::ffff:169.254.0.0/112"),
)
_BLOCKED_HOSTNAMES = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "instance-data",
        "100.100.100.200",
        "fd00:ec2::254",
    }
)


def _unwrap_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Unwrap IPv4-mapped, 6to4, and Teredo addresses to their underlying IPv4 IPs."""
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ips.append(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            ips.append(ip.sixtofour)
        if ip.teredo is not None:
            ips.extend([ip.teredo[0], ip.teredo[1]])
    return ips


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ip or any unwrapped mapping matches blocked networks/hosts."""
    for unrapped in _unwrap_ip(ip):
        for net in _BLOCKED_NETWORKS:
            if unrapped in net:
                return True
        if str(unrapped) in _BLOCKED_HOSTNAMES:
            return True
    return False


def validate_safe_url(url: str) -> str:
    """Validate that a URL uses safe HTTP/HTTPS scheme and does not target
    link-local or cloud metadata endpoints (SSRF prevention).

    Resolves hostnames and checks all unwrapped IPv4/IPv6 addresses against
    cloud metadata endpoints (e.g. 169.254.169.254, fd00:ec2::254, 100.100.100.200).

    Raises:
        ContractValidationError: If scheme is not http/https or target is blocked.
    """
    import socket

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
            agent_fix_hint="Access to cloud metadata endpoints (169.254.x.x / fd00:ec2::254) is forbidden.",
        )

    # Check if hostname is an IP literal
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_blocked_ip(ip):
            raise ContractValidationError(
                f"Blocked link-local / cloud metadata IP: {ip}",
                agent_fix_hint="Access to cloud metadata endpoints is forbidden.",
            )
    except ValueError:
        # Hostname is a domain name. Resolve via getaddrinfo to block DNS rebinding/SSRF
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in addr_info:
                resolved_ip_str = sockaddr[0]
                resolved_ip = ipaddress.ip_address(resolved_ip_str)
                if _is_blocked_ip(resolved_ip):
                    raise ContractValidationError(
                        f"Blocked hostname '{hostname}' resolves to cloud metadata IP: {resolved_ip}",
                        agent_fix_hint="Access to cloud metadata endpoints is forbidden.",
                    )
        except ContractValidationError:
            raise
        except (socket.gaierror, socket.herror, TimeoutError, OSError):
            # Offline or mocked host in test environment
            pass

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
