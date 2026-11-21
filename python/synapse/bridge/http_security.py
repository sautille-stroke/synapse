# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""HTTP security utilities: URL validation, SSRF prevention, and pinned-socket openers."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from synapse.errors import ContractValidationError

# Always-blocked link-local / cloud metadata / unspecified / multicast / NAT64 networks
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.100.100.200/32"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fd00:ec2::254/128"),
    ipaddress.ip_network("::ffff:169.254.0.0/112"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("ff00::/8"),
    # NAT64 well-known prefix (RFC 6052)
    ipaddress.ip_network("64:ff9b::/96"),
)

_LOOPBACK_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
)

_BLOCKED_HOSTNAMES = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "instance-data",
        "100.100.100.200",
        "fd00:ec2::254",
        "0.0.0.0",
        "::",
    }
)

_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


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


def _is_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool = False,
) -> bool:
    """Return True if ip or any unwrapped mapping matches blocked networks/hosts."""
    for unwrapped in _unwrap_ip(ip):
        if unwrapped.is_unspecified or unwrapped.is_multicast:
            return True
        for net in _ALWAYS_BLOCKED_NETWORKS:
            if unwrapped in net:
                return True
        if not allow_loopback:
            if unwrapped.is_loopback:
                return True
            for net in _LOOPBACK_NETWORKS:
                if unwrapped in net:
                    return True
        if str(unwrapped) in _BLOCKED_HOSTNAMES:
            return True
    return False


def validate_safe_url(
    url: str,
    *,
    allow_loopback: bool = False,
) -> tuple[str, list[str]]:
    """Validate that a URL uses safe HTTP/HTTPS scheme and does not target
    link-local, unspecified, multicast, or cloud metadata endpoints (SSRF prevention).

    Loopback addresses (``127.0.0.0/8``, ``::1``, ``localhost``) are blocked by
    default and permitted only when ``allow_loopback=True`` for trusted intra-pod
    sidecar services.

    DNS lookup failures are **fail-closed** (blocked).

    Returns:
        Tuple of ``(validated_url, resolved_ips)`` containing approved IP strings.

    Raises:
        ContractValidationError: If scheme is not http/https, DNS fails,
            or target is blocked.
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

    # Strip trailing dots from FQDN hostnames before validation
    hostname = hostname.rstrip(".")
    if not hostname:
        raise ContractValidationError(
            f"URL has no valid host after dot-stripping: {url}",
            agent_fix_hint="Check the URL format.",
        )

    if hostname in _BLOCKED_HOSTNAMES:
        raise ContractValidationError(
            f"Blocked link-local / cloud metadata / unspecified host: {hostname}",
            agent_fix_hint="Access to cloud metadata endpoints (169.254.x.x / fd00:ec2::254) and unspecified hosts is forbidden.",
        )
    if not allow_loopback and hostname in _LOOPBACK_HOSTNAMES:
        raise ContractValidationError(
            f"Blocked loopback host: {hostname}",
            agent_fix_hint="Loopback endpoints require explicit allow_loopback=True for trusted local sidecars.",
        )

    resolved_ips: list[str] = []

    # Check if hostname is an IP literal
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_blocked_ip(ip, allow_loopback=allow_loopback):
            raise ContractValidationError(
                f"Blocked link-local / cloud metadata / loopback IP: {ip}",
                agent_fix_hint="Access to cloud metadata endpoints and disallowed IPs is forbidden.",
            )
        resolved_ips.append(str(ip))
    except ValueError:
        # Hostname is a domain name. Resolve via getaddrinfo (fail-closed).
        try:
            addr_info = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        except (socket.gaierror, socket.herror, TimeoutError, OSError) as exc:
            raise ContractValidationError(
                f"DNS resolution failed for hostname '{hostname}': {exc}. "
                "Fail-closed: blocking request.",
                agent_fix_hint=(
                    f"Ensure hostname '{hostname}' resolves to a valid, "
                    "non-blocked IP address."
                ),
            ) from exc

        for _, _, _, _, sockaddr in addr_info:
            resolved_ip_str = str(sockaddr[0])
            resolved_ip = ipaddress.ip_address(resolved_ip_str)
            if _is_blocked_ip(resolved_ip, allow_loopback=allow_loopback):
                raise ContractValidationError(
                    f"Blocked hostname '{hostname}' resolves to blocked IP: {resolved_ip}",
                    agent_fix_hint="Access to cloud metadata endpoints, unspecified, and disallowed IPs is forbidden.",
                ) from None
            if resolved_ip_str not in resolved_ips:
                resolved_ips.append(resolved_ip_str)

        if not resolved_ips:
            raise ContractValidationError(
                f"DNS resolution returned no usable IP addresses for '{hostname}'.",
                agent_fix_hint="Verify DNS records for the target host.",
            ) from None

    return url, resolved_ips


def _connect_pinned_socket(
    host: str,
    port: int,
    timeout: float | None,
    source_address: tuple[str, int] | None,
    *,
    scheme: str,
    allow_loopback: bool,
) -> socket.socket:
    """Resolve, validate against SSRF, and connect directly to the approved IP."""
    probe_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    _, resolved_ips = validate_safe_url(
        f"{scheme}://{probe_host}:{port}/",
        allow_loopback=allow_loopback,
    )
    last_exc: OSError | None = None
    for ip_str in resolved_ips:
        try:
            return socket.create_connection(
                (ip_str, port),
                timeout=timeout,
                source_address=source_address,
            )
        except OSError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise OSError(f"Unable to connect to validated host '{host}:{port}'")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that pins the TCP dial to an SSRF-validated IP without rewriting Host."""

    allow_loopback: bool = False

    def connect(self) -> None:
        src_addr: tuple[str, int] | None = getattr(self, "source_address", None)
        self.sock = _connect_pinned_socket(
            self.host,
            self.port,
            self.timeout,
            src_addr,
            scheme="http",
            allow_loopback=self.allow_loopback,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that pins TCP dial to an SSRF-validated IP while preserving TLS SNI."""

    allow_loopback: bool = False

    def connect(self) -> None:
        import ssl

        src_addr: tuple[str, int] | None = getattr(self, "source_address", None)
        sock = _connect_pinned_socket(
            self.host,
            self.port,
            self.timeout,
            src_addr,
            scheme="https",
            allow_loopback=self.allow_loopback,
        )
        ctx: ssl.SSLContext = getattr(self, "_context", None) or ssl.create_default_context()
        # Preserve original hostname for TLS SNI and X.509 certificate verification
        self.sock = ctx.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, *, allow_loopback: bool = False) -> None:
        super().__init__()
        self._allow_loopback = allow_loopback

    def http_open(self, req: urllib.request.Request) -> Any:
        allow_lb = self._allow_loopback

        class _BoundHTTPConn(_PinnedHTTPConnection):
            allow_loopback = allow_lb

        return self.do_open(_BoundHTTPConn, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, *, allow_loopback: bool = False) -> None:
        super().__init__()
        self._allow_loopback = allow_loopback

    def https_open(self, req: urllib.request.Request) -> Any:
        allow_lb = self._allow_loopback
        ctx = getattr(self, "_context", None)

        class _BoundHTTPSConn(_PinnedHTTPSConnection):
            allow_loopback = allow_lb

        return self.do_open(_BoundHTTPSConn, req, context=ctx)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """HTTP redirect handler that validates target URLs against SSRF."""

    def __init__(self, *, allow_loopback: bool = False) -> None:
        super().__init__()
        self._allow_loopback = allow_loopback

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_safe_url(newurl, allow_loopback=self._allow_loopback)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_safe_opener(*, allow_loopback: bool = False) -> urllib.request.OpenerDirector:
    """Build a urllib opener with pinned-IP handlers, redirect guard, and proxies disabled."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedHTTPHandler(allow_loopback=allow_loopback),
        _PinnedHTTPSHandler(allow_loopback=allow_loopback),
        SafeRedirectHandler(allow_loopback=allow_loopback),
    )


def safe_urlopen(
    url_or_req: str | urllib.request.Request,
    timeout: float = 30.0,
    *,
    allow_loopback: bool = False,
) -> Any:
    """Open a URL or Request using pinned-IP connections and SSRF redirect guards.

    Preserves the full ``Request`` (including ``POST`` ``req.data``, ``Host``
    header, IPv6 framing, and HTTPS TLS SNI / certificate hostname verification)
    while binding TCP connections atomically to SSRF-validated IP addresses on
    both initial requests and HTTP redirects.
    """
    if isinstance(url_or_req, str):
        validate_safe_url(url_or_req, allow_loopback=allow_loopback)
        req = urllib.request.Request(url_or_req)
    else:
        validate_safe_url(url_or_req.full_url, allow_loopback=allow_loopback)
        req = url_or_req

    opener = build_safe_opener(allow_loopback=allow_loopback)
    return opener.open(req, timeout=timeout)
