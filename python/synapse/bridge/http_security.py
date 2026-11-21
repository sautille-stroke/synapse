# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""HTTP security utilities: URL validation, SSRF prevention, and safe openers."""

from __future__ import annotations

import ipaddress
import os
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
    # Loopback ranges
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    # NAT64 well-known prefix (RFC 6052)
    ipaddress.ip_network("64:ff9b::/96"),
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

# Proxy environment variables to suppress during URL validation
_PROXY_ENV_VARS = (
    "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
    "no_proxy", "NO_PROXY",
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
    for unwrapped in _unwrap_ip(ip):
        for net in _BLOCKED_NETWORKS:
            if unwrapped in net:
                return True
        if str(unwrapped) in _BLOCKED_HOSTNAMES:
            return True
    return False


def validate_safe_url(url: str) -> tuple[str, list[str]]:
    """Validate that a URL uses safe HTTP/HTTPS scheme and does not target
    link-local, loopback, or cloud metadata endpoints (SSRF prevention).

    Resolves hostnames and checks all unwrapped IPv4/IPv6 addresses against
    cloud metadata endpoints (e.g. 169.254.169.254, fd00:ec2::254, 100.100.100.200),
    loopback (127.x.x.x, ::1), and NAT64 prefixes (64:ff9b::/96).

    DNS lookup failures are **fail-closed** (blocked).

    Returns:
        Tuple of (validated_url, resolved_ips) where resolved_ips are the
        IP address strings from DNS resolution (empty for IP literals).

    Raises:
        ContractValidationError: If scheme is not http/https, DNS fails,
            or target is blocked.
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

    # Strip trailing dots from FQDN hostnames before validation
    hostname = hostname.rstrip(".")
    if not hostname:
        raise ContractValidationError(
            f"URL has no valid host after dot-stripping: {url}",
            agent_fix_hint="Check the URL format.",
        )

    if hostname in _BLOCKED_HOSTNAMES:
        raise ContractValidationError(
            f"Blocked link-local / cloud metadata / loopback host: {hostname}",
            agent_fix_hint="Access to cloud metadata endpoints (169.254.x.x / fd00:ec2::254) and loopback is forbidden.",
        )

    resolved_ips: list[str] = []

    # Check if hostname is an IP literal
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_blocked_ip(ip):
            raise ContractValidationError(
                f"Blocked link-local / cloud metadata / loopback IP: {ip}",
                agent_fix_hint="Access to cloud metadata endpoints and loopback is forbidden.",
            )
        resolved_ips.append(str(ip))
    except ValueError:
        # Hostname is a domain name. Resolve via getaddrinfo.
        # Temporarily suppress proxy environment variables during resolution
        # to prevent DNS-over-proxy attacks.
        saved_env: dict[str, str | None] = {}
        for var in _PROXY_ENV_VARS:
            saved_env[var] = os.environ.pop(var, None)
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except (socket.gaierror, socket.herror, TimeoutError, OSError) as exc:
            # DNS resolution failure: Fail-Closed — block the request
            raise ContractValidationError(
                f"DNS resolution failed for hostname '{hostname}': {exc}. "
                "Fail-closed: blocking request.",
                agent_fix_hint=(
                    f"Ensure hostname '{hostname}' resolves to a valid, "
                    "non-blocked IP address."
                ),
            ) from exc
        finally:
            # Restore proxy env vars
            for var, val in saved_env.items():
                if val is not None:
                    os.environ[var] = val

        for _, _, _, _, sockaddr in addr_info:
            resolved_ip_str = sockaddr[0]
            resolved_ip = ipaddress.ip_address(resolved_ip_str)
            if _is_blocked_ip(resolved_ip):
                raise ContractValidationError(
                    f"Blocked hostname '{hostname}' resolves to blocked IP: {resolved_ip}",
                    agent_fix_hint="Access to cloud metadata endpoints, loopback, and NAT64 is forbidden.",
                )
            resolved_ips.append(resolved_ip_str)

    return url, resolved_ips


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
    """Open a URL or Request using safe redirect handling and URL validation.

    Connects using the already-validated IP address to prevent DNS rebinding
    TOCTOU attacks: after ``validate_safe_url`` resolves and approves an IP,
    the connection is made directly to that IP (with the original ``Host``
    header) so a second DNS lookup cannot return a different address.
    """
    import socket
    import ssl

    if isinstance(url_or_req, str):
        _, resolved_ips = validate_safe_url(url_or_req)
        req = urllib.request.Request(url_or_req)
    else:
        _, resolved_ips = validate_safe_url(url_or_req.full_url)
        req = url_or_req

    # Pin connection to the validated IP to prevent DNS rebinding TOCTOU
    if resolved_ips:
        parsed = urllib.parse.urlparse(req.full_url)
        original_host = parsed.hostname or ""
        port = parsed.port
        validated_ip = resolved_ips[0]

        # Rewrite URL to use the validated IP directly
        if port:
            netloc = f"{validated_ip}:{port}"
        else:
            netloc = validated_ip
        pinned_url = urllib.parse.urlunparse(
            (parsed.scheme, netloc, parsed.path,
             parsed.params, parsed.query, parsed.fragment)
        )
        req = urllib.request.Request(
            pinned_url,
            headers=dict(req.headers),
            method=req.get_method(),
        )
        # Set Host header to original hostname for virtual hosting
        req.add_unredirected_header("Host", original_host)

    opener = build_safe_opener()
    return opener.open(req, timeout=timeout)
