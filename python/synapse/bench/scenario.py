"""Scenario data model and catalogue for Synapse-Bench.

A *Scenario* is the atomic unit of the Synapse-Bench evaluation dataset.
Each scenario specifies:

* An asset (procedural or downloadable)
* A camera trajectory type and parameters
* Rendering condition overrides
* Expected metric thresholds for pass/fail gating
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class ScenarioTier(str, Enum):
    """Scenario complexity tier."""

    T1_RIGID = "T1"  # Static rigid body, 360° orbit
    T2_DYNAMIC = "T2"  # Skeletal animation / physics sim
    T3_INTERACTION = "T3"  # Multi-body contact
    T4_STYLE = "T4"  # Local re-rendering + style alignment
    T5_E2E = "T5"  # End-to-end pipeline


class AssetSource(str, Enum):
    """Where the scenario asset comes from."""

    PROCEDURAL = "procedural"
    KHRONOS = "khronos"
    OBJAVERSE = "objaverse"
    LOCAL = "local"
    REGISTRY = "registry"


@dataclass
class Asset:
    """Reference to a 3D asset used in a scenario."""

    source: AssetSource
    id: str
    url: Optional[str] = None
    sha256: Optional[str] = None
    license: str = "MIT"

    def to_dict(self) -> dict:
        d = {"source": self.source.value, "id": self.id, "license": self.license}
        if self.url:
            d["url"] = self.url
        if self.sha256:
            d["sha256"] = self.sha256
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Asset":
        return cls(
            source=AssetSource(d["source"]),
            id=d["id"],
            url=d.get("url"),
            sha256=d.get("sha256"),
            license=d.get("license", "MIT"),
        )


@dataclass
class ExpectedMetrics:
    """Pass/fail thresholds for a scenario."""

    chamfer_mm: Optional[float] = None
    f_score_1mm: Optional[float] = None
    psnr_db: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    ciede2000: Optional[float] = None
    temporal_warp_error: Optional[float] = None
    flicker_score: Optional[float] = None
    foot_sliding_cm: Optional[float] = None
    bone_length_drift_pct: Optional[float] = None
    style_color_de: Optional[float] = None
    style_light_error_deg: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectedMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Scenario:
    """A single Synapse-Bench evaluation scenario.

    Attributes
    ----------
    temporal_mode:
        ``"continuous_clip"`` (default) for Wan 2.1 temporal video
        generation — enforces ``(num_frames - 1) % 4 == 0``.
        ``"discrete_views"`` for sparse multi-view (no temporal constraint).
    """

    id: str
    tier: ScenarioTier
    description: str = ""
    tags: list[str] = field(default_factory=list)

    asset: Asset = field(default_factory=lambda: Asset(AssetSource.PROCEDURAL, "cube"))
    camera_type: str = "orbit"
    camera_params: dict = field(default_factory=lambda: {
        "radius": 3.0,
        "elevation_deg": 15.0,
        "num_frames": 97,
        "fps": 24,
    })
    temporal_mode: Literal["continuous_clip", "discrete_views"] = "continuous_clip"

    conditions: dict = field(default_factory=dict)
    width: int = 1024
    height: int = 1024
    expected: ExpectedMetrics = field(default_factory=ExpectedMetrics)
    validates_stages: list[int] = field(default_factory=lambda: [2, 3])

    def __post_init__(self) -> None:
        """Validate T=4k+1 constraint for continuous clips."""
        num_frames = self.camera_params.get("num_frames", 120)
        if self.temporal_mode == "continuous_clip" and (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"Scenario '{self.id}': continuous_clip requires "
                f"(num_frames - 1) % 4 == 0, got num_frames={num_frames}. "
                f"Valid values: 5, 9, 13, ..., 97, 101, ..."
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "description": self.description,
            "tags": self.tags,
            "asset": self.asset.to_dict(),
            "camera_type": self.camera_type,
            "camera_params": self.camera_params,
            "temporal_mode": self.temporal_mode,
            "conditions": self.conditions,
            "width": self.width,
            "height": self.height,
            "expected": self.expected.to_dict(),
            "validates_stages": self.validates_stages,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(
            id=d["id"],
            tier=ScenarioTier(d["tier"]),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            asset=Asset.from_dict(d["asset"]),
            camera_type=d.get("camera_type", "orbit"),
            camera_params=d.get("camera_params", {}),
            temporal_mode=d.get("temporal_mode", "continuous_clip"),
            conditions=d.get("conditions", {}),
            width=d.get("width", 1024),
            height=d.get("height", 1024),
            expected=ExpectedMetrics.from_dict(d.get("expected", {})),
            validates_stages=d.get("validates_stages", [2, 3]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _scenarios_dir() -> Path:
    """Locate the scenarios directory.

    Primary: ``synapse.bench.scenarios`` package data (wheel-friendly).
    Fallback: ``data/scenarios/`` relative to repo root.
    """
    # Try package data first (importlib.resources)
    try:
        pkg_files = importlib.resources.files("synapse.bench") / "scenarios"
        pkg_path = Path(str(pkg_files))
        if pkg_path.is_dir():
            return pkg_path
    except (TypeError, ModuleNotFoundError):
        pass

    # Fallback to repo-relative data/scenarios/
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    return pkg_root / "data" / "scenarios"


def list_scenarios(tier: Optional[ScenarioTier] = None) -> list[Scenario]:
    """Return all built-in scenarios, optionally filtered by tier."""
    scenarios_dir = _scenarios_dir()
    if not scenarios_dir.is_dir():
        return []
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.json")):
        try:
            s = Scenario.load(path)
            if tier is None or s.tier == tier:
                scenarios.append(s)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return scenarios


def load_scenario(scenario_id: str) -> Scenario:
    """Load a built-in scenario by ID."""
    scenarios_dir = _scenarios_dir()
    path = scenarios_dir / f"{scenario_id}.json"
    if path.is_file():
        return Scenario.load(path)
    for p in scenarios_dir.glob("*.json"):
        s = Scenario.load(p)
        if s.id == scenario_id:
            return s
    raise FileNotFoundError(
        f"Scenario '{scenario_id}' not found in {scenarios_dir}"
    )
