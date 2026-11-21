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

from synapse.bench.scenario import (
    Scenario,
)
from synapse.bench.scenario import (
    list_scenarios as _list_scenarios,
)
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

    def resolve(self, name: str) -> Path:
        """Resolve an asset to a local file path.

        For builtin procedural assets, generates the mesh to an OBJ
        file in the cache directory. For remote assets, downloads with
        SHA-256 verification. Falls back to procedural for offline CI.

        Returns
        -------
        Path
            Absolute path to the local asset file.
        """
        entry = self.get(name)
        cache_dir = SYNAPSE_ASSET_CACHE
        cache_dir.mkdir(parents=True, exist_ok=True)

        if entry.builtin:
            return self._resolve_builtin(entry, cache_dir)

        if entry.url is not None:
            return self._resolve_remote(entry, cache_dir)

        # Fallback: attempt procedural generation
        if name in _BUILTIN_PROCEDURAL:
            return self._resolve_builtin(entry, cache_dir)

        raise FileNotFoundError(f"Cannot resolve asset '{name}': not builtin and no URL.")

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
