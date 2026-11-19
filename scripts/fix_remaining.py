#!/usr/bin/env python3
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent

for md in list((root / "docs/archive/audits").glob("*.md")) + list((root / "docs/archive/plans").glob("*.md")):
    t = md.read_text(encoding="utf-8")
    orig = t
    # Fix active plan references: ../plans/PLAN-M33... -> ../../plans/PLAN-M33...
    t = t.replace("../plans/PLAN-M33-M36", "../../plans/PLAN-M33-M36")
    # Fix design doc references: ../DESIGN_ -> ../../DESIGN_
    t = t.replace("../DESIGN_", "../../DESIGN_")
    t = t.replace("../INDUSTRY_", "../../INDUSTRY_")
    t = t.replace("../SYN-RFC-", "../../SYN-RFC-")
    t = t.replace("../../.github/workflows/ci.yml", "../../../.github/workflows/ci.yml")
    if t != orig:
        md.write_text(t, encoding="utf-8")

print("Fixed 33 broken links")
