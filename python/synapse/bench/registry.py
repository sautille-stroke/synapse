"""Asset and scenario registries for Synapse-Bench.

Provides :class:`AssetEntry`, :class:`AssetRegistry` for managing
benchmark assets (procedural and remote), and :class:`ScenarioRegistry`
for scanning and querying the built-in scenario catalogue.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from synapse.bench.scenario import (
    Asset,
    AssetSource,
    Scenario,
    validate_external_license,
)
from synapse.bench.scenario import (
    list_scenarios as _list_scenarios,
)
from synapse.errors import AssetNotFoundError, ContractValidationError
from synapse.render_core.procedural import (
    generate_procedural_mesh,
)

logger = logging.getLogger(__name__)

# Default cache directory for downloaded assets.
SYNAPSE_ASSET_CACHE: Path = Path(
    os.environ.get(
        "SYNAPSE_ASSET_CACHE",
        str(Path.home() / ".cache" / "synapse" / "assets"),
    )
)

# Built-in procedural asset names.
_BUILTIN_PROCEDURAL: set[str] = {
    "cube",
    "sphere",
    "cylinder",
    "torus",
    "plane",
    "cone",
    "humanoid_53",
}


@dataclass
class ExternalDatasetSpec:
    """Metadata descriptor for external 3D dataset ingestion & pre-validation."""

    source: AssetSource
    asset_id: str
    license: str = "CC-BY-4.0"
    dataset_category: str = ""
    sha256: str = ""
    fallback_procedural: str = "cube"
    up_axis: str = "Y"
    require_watertight: bool = True

    def __post_init__(self) -> None:
        validate_external_license(self.source, self.license)


def normalize_external_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    up_axis: str = "Z",
    target_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize an external 3D mesh to right-handed Y-up unit bounding sphere.

    1. Validates finite coordinates.
    2. Rotates ``Z-up -> Y-up`` via ``(x, y, z) -> (x, z, -y)`` when ``up_axis=='Z'``.
    3. Centers bounding-box centroid at origin ``(0, 0, 0)``.
    4. Scales uniformly to ``target_radius`` (preserving aspect ratio).
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ContractValidationError(
            "Cannot normalize empty or malformed mesh vertices.",
            agent_fix_hint="Provide vertices with non-empty shape (N, 3).",
        )
    if not np.all(np.isfinite(v)):
        raise ContractValidationError(
            "External mesh contains NaN or Inf vertex coordinates.",
            agent_fix_hint="Sanitize mesh geometry before normalization.",
        )

    axis_u = up_axis.strip().upper()
    if axis_u == "Z":
        # (x, y, z) -> (x, z, -y) right-handed determinant +1
        v = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    elif axis_u != "Y":
        raise ContractValidationError(
            f"Unsupported up_axis {up_axis!r}; expected 'Y' or 'Z'.",
            agent_fix_hint="Specify up_axis='Y' or up_axis='Z'.",
        )

    min_c = np.min(v, axis=0)
    max_c = np.max(v, axis=0)
    center = 0.5 * (min_c + max_c)
    v = v - center[np.newaxis, :]
    radii = np.linalg.norm(v, axis=1)
    max_r = float(np.max(radii))
    if max_r <= 1e-12:
        raise ContractValidationError(
            "Degenerate zero-extent mesh cannot be normalized to unit radius.",
            agent_fix_hint="Filter degenerate zero-volume assets in Stage 0.",
        )
    v = v * (float(target_radius) / max_r)
    return v.astype(np.float32), f.astype(np.int32)


def filter_stage0_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_faces: int = 4,
    max_faces: int = 2_000_000,
    min_area: float = 1e-12,
    require_manifold: bool = False,
    require_watertight: bool = False,
) -> tuple[bool, list[str]]:
    """Pre-ingestion Stage 0 topology & geometry filter for external 3D assets.

    Returns ``(passed, reasons)`` cleanly without throwing so callers can reject
    corrupt assets or fall back to procedural proxies.
    """
    reasons: list[str] = []
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        reasons.append("empty_or_invalid_vertices")
        return False, reasons
    if f.ndim != 2 or f.shape[1] < 3 or len(f) == 0:
        reasons.append("empty_or_invalid_faces")
        return False, reasons
    if not np.all(np.isfinite(v)):
        reasons.append("non_finite_vertex_coordinates")
    if len(f) < min_faces:
        reasons.append(f"face_count_below_min:{len(f)}<{min_faces}")
    if len(f) > max_faces:
        reasons.append(f"face_count_above_max:{len(f)}>{max_faces}")
    if int(np.min(f)) < 0 or int(np.max(f)) >= len(v):
        reasons.append("out_of_range_face_index")
        return False, reasons

    extent = np.max(v, axis=0) - np.min(v, axis=0)
    if float(np.max(extent)) <= 1e-8:
        reasons.append("zero_bounding_extent")

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    if int(np.sum(areas <= min_area)) > 0:
        reasons.append(f"degenerate_zero_area_faces:{int(np.sum(areas <= min_area))}")

    if require_manifold or require_watertight:
        from synapse.recon.geometry import check_manifold, check_watertight

        if require_manifold and not check_manifold(v, f):
            reasons.append("non_manifold_surface")
        if require_watertight and not check_watertight(v, f):
            reasons.append("non_watertight_surface")

    return len(reasons) == 0, reasons


@dataclass
class AssetEntry:
    """Descriptor for a benchmark asset.

    Attributes
    ----------
    name:
        Unique asset identifier.
    url:
        Optional remote URL for downloadable glTF / OBJ models.
    sha256:
        Expected SHA-256 hex digest of the downloaded file.
    builtin:
        ``True`` if the asset can be generated from procedural
        geometry without any external download.
    """

    name: str
    url: str | None = None
    sha256: str | None = None
    builtin: bool = False


class AssetRegistry:
    """Registry for benchmark assets.

    Pre-registers standard geometric primitives and the Humanoid-53
    proxy. Additional assets (remote glTF models) can be registered
    at runtime.
    """

    def __init__(self) -> None:
        self._entries: dict[str, AssetEntry] = {}
        self.last_resolution_meta: dict[str, Any] = {}
        self._register_builtins()

    # ── public API ────────────────────────────────────────────────

    def register(self, entry: AssetEntry) -> None:
        """Register an asset entry."""
        self._entries[entry.name] = entry

    def get(self, name: str) -> AssetEntry:
        """Retrieve an asset entry by name.

        Raises
        ------
        KeyError
            If the asset is not registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Asset '{name}' not registered. Available: {sorted(self._entries.keys())}"
            )
        return self._entries[name]

    def list_all(self) -> list[AssetEntry]:
        """Return all registered asset entries."""
        return list(self._entries.values())

    def resolve(
        self,
        name: str,
        *,
        fallback_procedural: str = "",
        strict_external: bool = False,
        expected_sha256: str = "",
        source: AssetSource | str | None = None,
    ) -> Path:
        """Resolve an asset to a local file path.

        For builtin procedural assets, generates the mesh to an OBJ
        file in the cache directory. For external assets, checks configured
        local external dataset roots first (with optional SHA-256 verification),
        then registered remote URLs, falling back to *fallback_procedural* for
        offline CI unless ``strict_external=True``.
        """
        cache_dir = SYNAPSE_ASSET_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_resolution_meta = {
            "asset_id": name,
            "source": source.value if isinstance(source, AssetSource) else str(source or ""),
            "fallback_used": False,
            "stage0_passed": True,
        }

        if name in self._entries:
            entry = self._entries[name]
            if entry.builtin:
                return self._resolve_builtin(entry, cache_dir)
            if entry.url is not None and not strict_external:
                path = self._resolve_remote(entry, cache_dir)
                return path

        # Check local external dataset directory (SYNAPSE_EXTERNAL_DATASETS_DIR / data/external)
        ext_roots: list[Path] = []
        env_dir = os.environ.get("SYNAPSE_EXTERNAL_DATASETS_DIR")
        if env_dir:
            ext_roots.append(Path(env_dir))
        open_data_dir = os.environ.get("DERVGEN_OPEN_DATA")
        if open_data_dir:
            ext_roots.append(Path(open_data_dir))
        repo_ext = Path(__file__).resolve().parents[3] / "data" / "external"
        ext_roots.append(repo_ext)

        src_sub = source.value if isinstance(source, AssetSource) else str(source or "")
        for root in ext_roots:
            if not root.exists():
                continue
            candidates = []
            for ext in (".obj", ".glb", ".ply", ".stl"):
                if src_sub:
                    candidates.append(root / src_sub / f"{name}{ext}")
                candidates.append(root / f"{name}{ext}")
            for cand in candidates:
                if cand.is_file():
                    if expected_sha256 and not self._verify_sha256(cand, expected_sha256):
                        raise ContractValidationError(
                            f"SHA256 mismatch for external asset {cand} (expected {expected_sha256}).",
                            agent_fix_hint="Re-stage or verify the integrity of the local external asset.",
                        )
                    self.last_resolution_meta["resolved_via"] = "local_external"
                    return cand

        if strict_external:
            raise AssetNotFoundError(
                f"External asset '{name}' not found locally and strict_external=True.",
                agent_fix_hint=(
                    "Set SYNAPSE_EXTERNAL_DATASETS_DIR to local dataset root or omit "
                    "strict_external to use fallback_procedural."
                ),
            )

        if fallback_procedural and fallback_procedural in _BUILTIN_PROCEDURAL:
            self.last_resolution_meta["fallback_used"] = True
            self.last_resolution_meta["resolved_via"] = f"fallback_procedural:{fallback_procedural}"
            return self._resolve_builtin(AssetEntry(name=fallback_procedural, builtin=True), cache_dir)

        if name in _BUILTIN_PROCEDURAL:
            return self._resolve_builtin(AssetEntry(name=name, builtin=True), cache_dir)

        entry = self.get(name)
        if entry.url is not None:
            return self._resolve_remote(entry, cache_dir)

        raise FileNotFoundError(f"Cannot resolve asset '{name}': not builtin and no URL.")

    def resolve_with_metadata(
        self,
        asset: Asset | str,
        *,
        strict_external: bool = False,
    ) -> tuple[Path, dict[str, Any]]:
        """Resolve a scenario :class:`Asset` or string ID and return ``(path, metadata)``."""
        if isinstance(asset, Asset):
            validate_external_license(asset.source, asset.license)
            path = self.resolve(
                asset.id,
                fallback_procedural=asset.fallback_procedural,
                strict_external=strict_external,
                expected_sha256=asset.sha256 or "",
                source=asset.source,
            )
            meta = dict(self.last_resolution_meta)
            meta["license"] = asset.license
            meta["up_axis"] = asset.up_axis
            meta["dataset_category"] = asset.dataset_category
            return path, meta

        path = self.resolve(asset, strict_external=strict_external)
        return path, dict(self.last_resolution_meta)

    # ── private ───────────────────────────────────────────────────

    def _register_builtins(self) -> None:
        """Pre-register standard procedural primitives and scanned assets."""
        for prim in _BUILTIN_PROCEDURAL:
            self._entries[prim] = AssetEntry(
                name=prim,
                builtin=True,
            )
        # Register standard GSO and Objaverse assets per SYN-RFC-0063
        scanned_assets = [
            (
                "gso_shoe",
                "https://raw.githubusercontent.com/google-research/google-research/master/scanned_objects/shoe.glb",
            ),
            (
                "gso_camera",
                "https://raw.githubusercontent.com/google-research/google-research/master/scanned_objects/camera.glb",
            ),
            (
                "objaverse_chair",
                "https://raw.githubusercontent.com/allenai/objaverse-xl/main/assets/chair.glb",
            ),
        ]
        for name, url in scanned_assets:
            self._entries[name] = AssetEntry(
                name=name,
                url=url,
                builtin=False,
            )

    def _resolve_builtin(
        self,
        entry: AssetEntry,
        cache_dir: Path,
    ) -> Path:
        """Generate a procedural mesh and cache as OBJ."""
        obj_path = cache_dir / f"{entry.name}.obj"
        if obj_path.is_file():
            return obj_path

        if entry.name == "humanoid_53":
            from synapse.render_core.procedural import (
                generate_humanoid_proxy,
            )

            verts, faces, _ = generate_humanoid_proxy()
            # Write simple OBJ
            lines = [
                "# Synapse-Bench humanoid_53 proxy",
                f"# Vertices: {len(verts)}, Faces: {len(faces)}",
            ]
            for v in verts:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            for f in faces:
                lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            return obj_path

        if entry.name == "cone":
            mesh = generate_procedural_mesh("cone")
            mesh.save_obj(obj_path)
            return obj_path

        # Standard procedural types
        mesh = generate_procedural_mesh(entry.name)
        mesh.save_obj(obj_path)
        return obj_path

    def _resolve_remote(
        self,
        entry: AssetEntry,
        cache_dir: Path,
    ) -> Path:
        """Download a remote asset with SHA-256 verification."""
        assert entry.url is not None
        if entry.url.lower().startswith("file://"):
            from synapse.errors import ContractValidationError

            raise ContractValidationError(
                f"file:// URLs forbidden in asset registry: {entry.url}",
                agent_fix_hint="Use http:// or https:// URLs for remote benchmark assets.",
            )

        # Hash bucketing to avoid collisions
        url_hash = hashlib.sha256(entry.url.encode("utf-8")).hexdigest()[:16]
        filename = entry.url.rsplit("/", 1)[-1]
        local_path = cache_dir / f"{url_hash}_{filename}"

        if local_path.is_file():
            if entry.sha256 and not self._verify_sha256(
                local_path,
                entry.sha256,
            ):
                logger.warning(
                    "SHA256 mismatch for cached %s, re-downloading",
                    local_path,
                )
                local_path.unlink()
            else:
                return local_path

        # Download with timeout and clean up partial file on failure
        try:
            from synapse.bridge.http_security import safe_urlopen

            logger.info("Downloading %s -> %s", entry.url, local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with safe_urlopen(entry.url, timeout=30.0) as resp, open(local_path, "wb") as f:
                while chunk := resp.read(65536):
                    f.write(chunk)
        except Exception as exc:
            if local_path.is_file():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            # Offline fallback: try procedural
            fallback_name = entry.name
            if fallback_name.startswith("gso_") or fallback_name.startswith("objaverse_"):
                fallback_name = (
                    "cube" if "chair" in fallback_name or "camera" in fallback_name else "cylinder"
                )
            if fallback_name in _BUILTIN_PROCEDURAL or entry.name in _BUILTIN_PROCEDURAL:
                logger.warning(
                    "Download failed for %s, falling back to procedural %s: %s",
                    entry.name,
                    fallback_name,
                    exc,
                )
                return self._resolve_builtin(
                    AssetEntry(name=fallback_name, builtin=True), cache_dir
                )
            raise FileNotFoundError(
                f"Cannot download asset '{entry.name}' from {entry.url}: {exc}"
            ) from exc

        # Verify checksum
        if entry.sha256 and not self._verify_sha256(
            local_path,
            entry.sha256,
        ):
            if local_path.is_file():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            raise ValueError(f"SHA256 mismatch for {entry.name}: expected {entry.sha256}")

        return local_path

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> bool:
        """Verify SHA-256 digest of a file."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest() == expected


class ScenarioRegistry:
    """Registry that scans and indexes built-in benchmark scenarios.

    On initialisation, scans ``python/synapse/bench/scenarios/*.json``
    and indexes all scenarios by ID and tier.
    """

    def __init__(self) -> None:
        self._scenarios: dict[str, Scenario] = {}
        self._by_tier: dict[str, list[str]] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan scenario directory and build indices."""
        all_scenarios = _list_scenarios()
        for s in all_scenarios:
            self._scenarios[s.id] = s
            tier_key = s.tier.value
            self._by_tier.setdefault(tier_key, []).append(s.id)

    def list_scenarios(self) -> list[str]:
        """Return all scenario IDs in sorted order."""
        return sorted(self._scenarios.keys())

    def load_scenario(self, name: str) -> Scenario:
        """Load a scenario by ID.

        Raises
        ------
        KeyError
            If the scenario ID is not found.
        """
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found. Available: {self.list_scenarios()}")
        return self._scenarios[name]

    def list_by_tier(self, tier: str) -> list[str]:
        """Return scenario IDs filtered by tier (e.g. 'T1').

        Parameters
        ----------
        tier:
            Tier string such as ``"T1"``, ``"T2"``, etc.
        """
        return sorted(self._by_tier.get(tier, []))
