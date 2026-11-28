#!/usr/bin/env python3
"""Synchronization validator against ecosystem Ruff SSOT (PR-212).

Safe replacement for adding brittle relative ``extend = "../rules_dervgen_env/config/ruff.toml"``
directly in ``pyproject.toml`` (which breaks standalone clones & public GitHub Actions).
Checks parity against ``../rules_dervgen_env/config/ruff.toml`` whenever present, and
exits cleanly on single-repo checkouts unless ``--require-ssot`` is specified.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import sys
from pathlib import Path
from typing import Any


def _parse_toml_fallback(text: str) -> dict[str, Any]:
    """Minimal TOML subset parser for Python <3.11 without tomllib/tomli."""
    root: dict[str, Any] = {}
    current = root
    raw_lines = text.splitlines()
    idx = 0
    while idx < len(raw_lines):
        line = raw_lines[idx].split("#", 1)[0].strip()
        idx += 1
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip()
            target = root
            for part in sec.split("."):
                nxt = target.setdefault(part.strip(), {})
                if not isinstance(nxt, dict):
                    nxt = {}
                    target[part.strip()] = nxt
                target = nxt
            current = target
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            key = k.strip().strip('"').strip("'")
            val_s = v.strip()
            while val_s.count("[") > val_s.count("]") and idx < len(raw_lines):
                nxt_line = raw_lines[idx].split("#", 1)[0].strip()
                val_s += " " + nxt_line
                idx += 1
            if val_s in ("true", "false"):
                current[key] = val_s == "true"
            else:
                try:
                    current[key] = ast.literal_eval(val_s)
                except Exception:
                    current[key] = val_s
    return root


def _loads_toml(text: str) -> dict[str, Any]:
    """Load TOML using stdlib tomllib (3.11+), tomli, or pure-Python fallback."""
    for mod_name in ("tomllib", "tomli"):
        try:
            mod: Any = importlib.import_module(mod_name)
            return dict(mod.loads(text))
        except ImportError:
            continue
    return _parse_toml_fallback(text)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DEFAULT_SSOT_PROBES = [
    _REPO_ROOT.parent / "rules_dervgen_env" / "config" / "ruff.toml",
    Path("../rules_dervgen_env/config/ruff.toml"),
]


def _extract_ruff_config(data: dict[str, Any], *, is_pyproject: bool) -> dict[str, Any]:
    """Normalize Ruff configuration keys from pyproject.toml or standalone ruff.toml."""
    if is_pyproject:
        tool_cfg = data.get("tool", {})
        ruff_cfg = tool_cfg.get("ruff", {}) if isinstance(tool_cfg, dict) else {}
    else:
        ruff_cfg = data
    if not isinstance(ruff_cfg, dict):
        return {}
    lint_cfg = ruff_cfg.get("lint", {})
    select = lint_cfg.get("select", []) if isinstance(lint_cfg, dict) else []
    return {
        "target_version": str(ruff_cfg.get("target-version", "")),
        "line_length": int(ruff_cfg.get("line-length", 0)) if ruff_cfg.get("line-length") else 0,
        "select": [str(item) for item in select] if isinstance(select, list) else [],
        "extend": str(ruff_cfg.get("extend", "")),
    }


def check_ruff_ssot_parity(
    pyproject_path: Path = _DEFAULT_PYPROJECT,
    *,
    ssot_path: Path | None = None,
    require_ssot: bool = False,
) -> tuple[bool, list[str], bool]:
    """Verify ``pyproject.toml`` Ruff config is standalone and aligned with SSOT if present.

    Returns:
        ``(ok, messages, ssot_found)``.
    """
    messages: list[str] = []
    if not pyproject_path.is_file():
        return False, [f"FAIL: pyproject.toml not found at {pyproject_path}"], False

    py_data = _loads_toml(pyproject_path.read_text(encoding="utf-8"))
    local_cfg = _extract_ruff_config(py_data, is_pyproject=True)

    # Guard against brittle relative extend inside standalone pyproject.toml
    if local_cfg.get("extend"):
        messages.append(
            f"FAIL: pyproject.toml defines tool.ruff.extend={local_cfg['extend']!r}, "
            "which breaks standalone single-repo CI runners."
        )
        return False, messages, False

    resolved_ssot: Path | None = None
    if ssot_path is not None:
        if ssot_path.is_file():
            resolved_ssot = ssot_path
    else:
        for cand in _DEFAULT_SSOT_PROBES:
            if cand.is_file():
                resolved_ssot = cand
                break

    if resolved_ssot is None:
        if require_ssot:
            messages.append(
                "FAIL: --require-ssot set, but ../rules_dervgen_env/config/ruff.toml not found."
            )
            return False, messages, False
        messages.append("SKIPPED: upstream ../rules_dervgen_env/config/ruff.toml not present.")
        return True, messages, False

    ssot_data = _loads_toml(resolved_ssot.read_text(encoding="utf-8"))
    ssot_cfg = _extract_ruff_config(ssot_data, is_pyproject=False)

    errors: list[str] = []
    if ssot_cfg["target_version"] and local_cfg["target_version"] != ssot_cfg["target_version"]:
        errors.append(
            f"target-version drift: local={local_cfg['target_version']!r} vs ssot={ssot_cfg['target_version']!r}"
        )
    if ssot_cfg["line_length"] and local_cfg["line_length"] != ssot_cfg["line_length"]:
        errors.append(
            f"line-length drift: local={local_cfg['line_length']} vs ssot={ssot_cfg['line_length']}"
        )
    def _is_rule_covered(rule: str, selected: list[str]) -> bool:
        for sel in selected:
            if rule == sel:
                return True
            if sel.isalpha() and rule.startswith(sel) and rule[len(sel) :].isdigit():
                return True
        return False

    missing_rules = [
        r for r in ssot_cfg["select"] if not _is_rule_covered(r, local_cfg["select"])
    ]
    if missing_rules:
        errors.append(f"Missing SSOT lint.select rule(s) in pyproject.toml: {missing_rules}")

    if errors:
        return False, [f"FAIL: {err}" for err in errors], True
    return True, [f"OK: pyproject.toml Ruff config aligned with {resolved_ssot}"], True


def sync_ruff_ssot(
    pyproject_path: Path = _DEFAULT_PYPROJECT,
    *,
    ssot_path: Path,
) -> bool:
    """Sync baseline ``target-version`` and ``line-length`` from SSOT into ``pyproject.toml``."""
    if not pyproject_path.is_file() or not ssot_path.is_file():
        return False
    ssot_data = _loads_toml(ssot_path.read_text(encoding="utf-8"))
    ssot_cfg = _extract_ruff_config(ssot_data, is_pyproject=False)
    text = pyproject_path.read_text(encoding="utf-8")

    if ssot_cfg["target_version"]:
        text = re.sub(
            r'(?m)^(target-version\s*=\s*)"[^"]*"',
            rf'\1"{ssot_cfg["target_version"]}"',
            text,
            count=1,
        )
    if ssot_cfg["line_length"]:
        text = re.sub(
            r"(?m)^(line-length\s*=\s*)\d+",
            rf"\g<1>{ssot_cfg['line_length']}",
            text,
            count=1,
        )
    pyproject_path.write_text(text, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Ruff SSOT parity check/sync."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit 1 on drift or violations.")
    parser.add_argument("--sync", action="store_true", help="Sync baseline SSOT settings locally.")
    parser.add_argument(
        "--require-ssot",
        action="store_true",
        help="Fail closed if ../rules_dervgen_env/config/ruff.toml is absent.",
    )
    parser.add_argument("--ssot-path", type=Path, default=None, help="Explicit SSOT ruff.toml path.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=_DEFAULT_PYPROJECT,
        help="Path to synapse pyproject.toml.",
    )
    args = parser.parse_args(argv)

    if args.sync:
        target_ssot = args.ssot_path or _DEFAULT_SSOT_PROBES[0]
        if not sync_ruff_ssot(args.pyproject, ssot_path=target_ssot):
            print(f"FAIL: Could not sync from {target_ssot}", file=sys.stderr)
            return 1
        print(f"Synced baseline Ruff settings from {target_ssot} -> {args.pyproject}")
        return 0

    ok, messages, _ = check_ruff_ssot_parity(
        args.pyproject,
        ssot_path=args.ssot_path,
        require_ssot=args.require_ssot,
    )
    for msg in messages:
        print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
