#!/usr/bin/env python3
"""Scan all Markdown files for broken relative links."""

from __future__ import annotations

import pathlib
import re
import sys


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    ignored = {"bazel-", "node_modules", ".git", "venv", ".pytest_cache"}
    md_files = [
        f
        for f in root.rglob("*.md")
        if not any(any(ign in p for ign in ignored) for p in f.parts)
    ]
    broken: list[str] = []
    link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

    for md in md_files:
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in link_re.finditer(text):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "#", "mailto:", "javascript:")):
                continue
            # Strip anchor
            target_path = target.split("#")[0]
            if not target_path:
                continue
            resolved = (md.parent / target_path).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(root)}: {target}")

    if broken:
        print(f"Found {len(broken)} broken local link(s):")
        for b in broken[:30]:
            print(f"  {b}")
        if len(broken) > 30:
            print(f"  ... and {len(broken) - 30} more")
        return 1

    print(f"Scanned {len(md_files)} markdown files, all local links OK (0 broken).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
