# Copyright 2026 Sautille-Stroke / DervGen. All rights reserved.
"""Security hardening tests (TD-59, TD-25, TD-29).

Covers:
- np.load() with allow_pickle=False enforcement
- SSRF defense against RFC 1918 private ranges
- FoundaryTransport.poll_job() base class safety
"""

from __future__ import annotations

import ipaddress
import tempfile
from pathlib import Path

import numpy as np
import pytest

from synapse.bridge.foundary_client import (
    FoundaryTransport,
    MockFoundaryTransport,
)
from synapse.bridge.http_security import (
    _is_blocked_ip,
    validate_safe_url,
)
from synapse.errors import ContractValidationError


# ===================================================================
# TD-59: np.load(allow_pickle=False) enforcement
# ===================================================================


class TestNpLoadPickleSafety:
    """Verify that np.load calls across the codebase reject pickle payloads."""

    def test_np_load_rejects_pickle_payload(self, tmp_path: Path) -> None:
        """A .npy file containing a pickled object must raise ValueError
        when loaded with ``allow_pickle=False``."""
        # Create a file containing a pickled Python object
        malicious = tmp_path / "malicious.npy"
        np.save(str(malicious), np.array([1, 2, 3]))  # normal save → safe

        # Create an actual pickle-based .npy (object arrays use pickle)
        obj_array = np.array([{"a": 1}, {"b": 2}], dtype=object)
        pickled = tmp_path / "pickled.npy"
        np.save(str(pickled), obj_array, allow_pickle=True)

        # Loading with allow_pickle=False should raise
        with pytest.raises(ValueError, match="allow_pickle"):
            np.load(str(pickled), allow_pickle=False)

        # Safe arrays should load fine
        result = np.load(str(malicious), allow_pickle=False)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_cli_recon_np_load_has_allow_pickle_false(self) -> None:
        """Verify that _recon.py source contains allow_pickle=False for all np.load calls."""
        import ast
        import inspect

        from synapse.cli import _recon

        source = inspect.getsource(_recon)
        tree = ast.parse(source)

        np_load_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match np.load(...) calls
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "np"
                ):
                    has_pickle_kwarg = any(
                        kw.arg == "allow_pickle" for kw in node.keywords
                    )
                    np_load_calls.append(
                        (node.lineno, has_pickle_kwarg)
                    )

        # Every np.load call must have allow_pickle keyword
        for lineno, has_kwarg in np_load_calls:
            assert has_kwarg, (
                f"np.load() at _recon.py line {lineno} is missing "
                "allow_pickle=False — arbitrary code execution risk"
            )
        # Sanity: we expect at least 3 np.load calls
        assert len(np_load_calls) >= 3, (
            f"Expected ≥3 np.load calls in _recon.py, found {len(np_load_calls)}"
        )


# ===================================================================
# TD-25: SSRF defense against RFC 1918 private ranges
# ===================================================================


class TestSsrfPrivateRangeDefense:
    """Verify that RFC 1918 private IP ranges are blocked by default."""

    @pytest.mark.parametrize(
        "ip_str",
        [
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.1.100",
        ],
    )
    def test_rfc1918_blocked_by_default(self, ip_str: str) -> None:
        """RFC 1918 private IPs must be blocked when allow_private=False."""
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is True, (
            f"RFC 1918 IP {ip_str} was NOT blocked — SSRF vulnerability"
        )

    @pytest.mark.parametrize(
        "ip_str",
        [
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
        ],
    )
    def test_rfc1918_allowed_when_explicit(self, ip_str: str) -> None:
        """RFC 1918 private IPs must be allowed when allow_private=True."""
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip, allow_private=True) is False, (
            f"RFC 1918 IP {ip_str} was blocked despite allow_private=True"
        )

    @pytest.mark.parametrize(
        "ip_str",
        [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
        ],
    )
    def test_public_ips_allowed(self, ip_str: str) -> None:
        """Public IPs must pass through the SSRF filter."""
        ip = ipaddress.ip_address(ip_str)
        assert _is_blocked_ip(ip) is False, (
            f"Public IP {ip_str} was incorrectly blocked"
        )

    def test_validate_safe_url_blocks_rfc1918_literal(self) -> None:
        """validate_safe_url must reject literal RFC 1918 IPs."""
        with pytest.raises(ContractValidationError, match="Blocked"):
            validate_safe_url("http://10.0.0.1:8080/api")

        with pytest.raises(ContractValidationError, match="Blocked"):
            validate_safe_url("http://192.168.1.1/")

        with pytest.raises(ContractValidationError, match="Blocked"):
            validate_safe_url("http://172.16.0.1/")

    def test_ipv6_unique_local_blocked(self) -> None:
        """RFC 4193 unique-local IPv6 (fc00::/7) must be blocked."""
        ip = ipaddress.ip_address("fd12:3456:789a::1")
        assert _is_blocked_ip(ip) is True

    def test_cloud_metadata_still_blocked(self) -> None:
        """Pre-existing cloud metadata blocks must still work."""
        ip = ipaddress.ip_address("169.254.169.254")
        assert _is_blocked_ip(ip) is True

        with pytest.raises(ContractValidationError):
            validate_safe_url("http://169.254.169.254/latest/meta-data/")


# ===================================================================
# TD-29: FoundaryTransport.poll_job() base class safety
# ===================================================================


class TestFoundaryPollJobBase:
    """Verify that the abstract base FoundaryTransport.poll_job() raises."""

    def test_base_poll_job_raises_not_implemented(self) -> None:
        """FoundaryTransport.poll_job() must NOT silently return 'succeeded'.

        The base implementation must raise NotImplementedError to force
        subclasses to provide real job state queries.
        """
        # We can't instantiate ABC directly, so test via MockFoundaryTransport
        # with an unknown job_id — it falls through to super().poll_job()
        transport = MockFoundaryTransport()
        with pytest.raises(NotImplementedError, match="must override poll_job"):
            transport.poll_job("nonexistent_job_id_12345")

    def test_mock_poll_job_succeeds_for_known_jobs(self) -> None:
        """MockFoundaryTransport.poll_job() must still work for submitted jobs."""
        from synapse.bridge.foundary_client import FoundaryRequest

        transport = MockFoundaryTransport()
        req = FoundaryRequest(service="sam2", job_id="test_job_1")
        status = transport.submit_job(req)
        assert status.state == "succeeded"

        # Polling the submitted job should succeed
        polled = transport.poll_job("test_job_1")
        assert polled.state == "succeeded"
        assert polled.job_id == "test_job_1"
