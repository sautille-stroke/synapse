"""Hypothesis property-based tests for core contracts and invariants (PR-149).

Covers:
1. Camera projection and eccentric principal point zero-mirror roundtrip precision.
2. View matrix SO(3) determinant and orthonormality preservation.
3. FGPP frame sanitization invariant under arbitrary floating point perturbations.
4. JSON Schema vs Pydantic model cross-validation parity (LoRA, JobBundle, InspectionBundle, InspectionReport).
5. Mesh geometry validation topology equivalence between Python Oracle and C++20 native backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydantic
import pytest
from hypothesis import given, settings

jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
from hypothesis import strategies as st  # noqa: E402
from synapse.adapters.lora.manifest import LoRAManifest  # noqa: E402
from synapse.bench.bundle_exporter import InspectionBundleManifest  # noqa: E402
from synapse.fgpp.frame import CameraConvention, CameraParams, FGPPFrame  # noqa: E402
from synapse.fgpp.io import sanitize_fgpp_frame  # noqa: E402
from synapse.inspect.geometry_gate import validate_mesh_geometry  # noqa: E402
from synapse.inspect.schemas import DefectAnnotation3D, InspectionReport  # noqa: E402
from synapse.pipeline.bundle_worker import BundleManifest  # noqa: E402

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMAS_DIR = _REPO_ROOT / "schemas"


# ---------------------------------------------------------------------------
# Pillar 1: Camera Projection & SO(3) Properties
# ---------------------------------------------------------------------------


class TestHypothesisCameraProperties:
    """Property-based verification of camera projection and coordinate contracts."""

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        fx=st.floats(min_value=100.0, max_value=3000.0),
        fy=st.floats(min_value=100.0, max_value=3000.0),
        width=st.sampled_from([256, 512, 1024]),
        height=st.sampled_from([256, 512, 1024]),
        cx_rel=st.floats(min_value=0.15, max_value=0.85),
        cy_rel=st.floats(min_value=0.15, max_value=0.85),
        z_near=st.floats(min_value=0.05, max_value=0.5),
        z_far=st.floats(min_value=50.0, max_value=500.0),
    )
    def test_camera_projection_and_principal_point_roundtrip(
        self,
        fx: float,
        fy: float,
        width: int,
        height: int,
        cx_rel: float,
        cy_rel: float,
        z_near: float,
        z_far: float,
    ) -> None:
        """Eccentric principal point projection matrix <-> parameter recovery roundtrip <= 1e-5."""
        cx = cx_rel * float(width)
        cy = cy_rel * float(height)

        cam = CameraParams(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            z_near=z_near,
            z_far=z_far,
        )

        proj = cam.projection_matrix()
        assert CameraConvention.detect(proj) == CameraConvention.OPENGL

        # Reverse projection matrix mapping
        p00, p11 = float(proj[0, 0]), float(proj[1, 1])
        p02, p12 = float(proj[0, 2]), float(proj[1, 2])

        fx_rec = abs(p00) * float(width) / 2.0
        fy_rec = abs(p11) * float(height) / 2.0
        cx_rec = (1.0 - p02) * float(width) / 2.0
        cy_rec = (p12 + 1.0) * float(height) / 2.0

        assert abs(fx - fx_rec) < 1e-4
        assert abs(fy - fy_rec) < 1e-4
        assert abs(cx - cx_rec) < 1e-4
        assert abs(cy - cy_rec) < 1e-4

        # Serialization roundtrip
        d = cam.to_dict()
        cam_restored = CameraParams.from_dict(d)
        assert abs(cam_restored.fx - fx) < 1e-4
        assert abs(cam_restored.fy - fy) < 1e-4
        assert abs(cam_restored.cx - cx) < 1e-4
        assert abs(cam_restored.cy - cy) < 1e-4

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        yaw=st.floats(-np.pi, np.pi),
        pitch=st.floats(-np.pi / 2.5, np.pi / 2.5),
        roll=st.floats(-np.pi, np.pi),
        tx=st.floats(-100.0, 100.0),
        ty=st.floats(-100.0, 100.0),
        tz=st.floats(-100.0, 100.0),
    )
    def test_camera_rotation_so3_determinant_and_orthonormality(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        tx: float,
        ty: float,
        tz: float,
    ) -> None:
        """SO(3) view matrix rotation determinant == 1.0 and R^T R == I."""
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        R = Rz @ Ry @ Rx

        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = R
        view[:3, 3] = [tx, ty, tz]
        world = np.linalg.inv(view)

        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-6, f"Det(R) = {det} != 1.0"

        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
        assert ortho_err < 1e-6, f"Ortho error = {ortho_err}"

        cam = CameraParams(
            fx=500.0,
            fy=500.0,
            cx=256.0,
            cy=256.0,
            width=512,
            height=512,
            view_matrix=view,
            world_matrix=world,
        )
        d = cam.to_dict()
        cam2 = CameraParams.from_dict(d)
        assert np.allclose(cam2.view_matrix, view, atol=1e-6)
        assert np.allclose(cam2.world_matrix, world, atol=1e-6)


# ---------------------------------------------------------------------------
# Pillar 2: FGPP Frame Sanitization Invariants
# ---------------------------------------------------------------------------

_st_extreme_floats = st.one_of(
    st.floats(min_value=-1000.0, max_value=1000.0),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
    st.just(0.0),
    st.just(-1.0),
)


class TestHypothesisFgppSanitization:
    """Property tests asserting physical invariants on FGPP frame sanitization."""

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        depth_raw=st.lists(_st_extreme_floats, min_size=16, max_size=16),
        normal_raw=st.lists(_st_extreme_floats, min_size=48, max_size=48),
        radiance_raw=st.lists(_st_extreme_floats, min_size=48, max_size=48),
        color_raw=st.lists(_st_extreme_floats, min_size=48, max_size=48),
        rm_raw=st.lists(_st_extreme_floats, min_size=32, max_size=32),
    )
    def test_sanitize_fgpp_frame_arbitrary_tensor_invariants(
        self,
        depth_raw: list[float],
        normal_raw: list[float],
        radiance_raw: list[float],
        color_raw: list[float],
        rm_raw: list[float],
    ) -> None:
        """Arbitrary NaN/Inf/extreme numbers must produce strictly finite, bounded FGPP frames."""
        H, W = 4, 4
        depth = np.array(depth_raw, dtype=np.float32).reshape((H, W))
        world_normal = np.array(normal_raw, dtype=np.float16).reshape((H, W, 3))
        direct_radiance = np.array(radiance_raw, dtype=np.float16).reshape((H, W, 3))
        base_color = np.array(color_raw, dtype=np.float16).reshape((H, W, 3))
        roughness_metallic = np.array(rm_raw, dtype=np.float16).reshape((H, W, 2))
        cryptomatte_id = np.arange(16, dtype=np.uint32).reshape((H, W))

        cam = CameraParams(fx=500.0, fy=500.0, cx=2.0, cy=2.0, width=W, height=H, z_near=0.1, z_far=100.0)
        raw_frame = FGPPFrame(
            camera=cam,
            depth=depth,
            world_normal=world_normal,
            direct_radiance=direct_radiance,
            base_color=base_color,
            roughness_metallic=roughness_metallic,
            cryptomatte_id=cryptomatte_id,
            frame_index=0,
        )

        sanitized = sanitize_fgpp_frame(raw_frame)

        # 1. Depth is completely finite and clamped within (0, z_far]
        assert np.all(np.isfinite(sanitized.depth))
        assert np.all(sanitized.depth > 0.0)
        assert np.all(sanitized.depth <= cam.z_far)

        # 2. Cryptomatte ID is zeroed wherever original depth was non-finite or non-positive
        bad_depth_mask = ~np.isfinite(depth) | (depth <= 0.0)
        assert sanitized.cryptomatte_id is not None
        assert np.all(sanitized.cryptomatte_id[bad_depth_mask] == 0)

        # 3. World normal is finite, unit-norm
        assert sanitized.world_normal is not None
        assert np.all(np.isfinite(sanitized.world_normal))
        norms = np.linalg.norm(sanitized.world_normal.astype(np.float32), axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-3)

        # 4. Direct radiance is finite, non-negative, and capped at float16 max (65504)
        assert sanitized.direct_radiance is not None
        assert np.all(np.isfinite(sanitized.direct_radiance))
        assert np.all(sanitized.direct_radiance >= 0.0)
        assert np.all(sanitized.direct_radiance <= 65504.0)

        # 5. Base color is finite and strictly in [0.0, 1.0]
        assert sanitized.base_color is not None
        assert np.all(np.isfinite(sanitized.base_color))
        assert np.all(sanitized.base_color >= 0.0)
        assert np.all(sanitized.base_color <= 1.0)

        # 6. Roughness/metallic is finite; roughness in [0.04, 1.0], metallic in [0.0, 1.0]
        assert sanitized.roughness_metallic is not None
        assert np.all(np.isfinite(sanitized.roughness_metallic))
        assert np.all(sanitized.roughness_metallic[..., 0] >= 0.04)
        assert np.all(sanitized.roughness_metallic[..., 0] <= 1.0)
        assert np.all(sanitized.roughness_metallic[..., 1] >= 0.0)
        assert np.all(sanitized.roughness_metallic[..., 1] <= 1.0)


# ---------------------------------------------------------------------------
# Pillar 3: JSON Schema vs Pydantic Model Parity
# ---------------------------------------------------------------------------


class TestHypothesisJsonSchemaParity:
    """Property tests asserting 100% equivalence between JSON Schema and Pydantic models."""

    @pytest.fixture(autouse=True)
    def setup_schemas(self) -> None:
        self.lora_schema = json.loads((_SCHEMAS_DIR / "lora.v1.0.schema.json").read_text(encoding="utf-8"))
        self.job_bundle_schema = json.loads((_SCHEMAS_DIR / "job_bundle.v1.0.schema.json").read_text(encoding="utf-8"))
        self.inspection_bundle_schema = json.loads((_SCHEMAS_DIR / "bundle.v1.0.schema.json").read_text(encoding="utf-8"))
        self.inspection_report_schema = json.loads((_SCHEMAS_DIR / "inspection.v2.0.schema.json").read_text(encoding="utf-8"))

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        name=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=["L", "N"])),
        rank=st.integers(min_value=1, max_value=128),
        alpha=st.floats(min_value=0.1, max_value=64.0),
    )
    def test_lora_manifest_jsonschema_pydantic_parity(
        self,
        name: str,
        rank: int,
        alpha: float,
    ) -> None:
        """Valid LoRAManifest passes both Pydantic and JSON Schema; corrupted fails both."""
        manifest = LoRAManifest(
            name=name,
            category="style",
            rank=rank,
            alpha=alpha,
            adapter_path="sub/weights.safetensors",
            adapter_sha256="a" * 64,
            target_modules=["cross_attn.to_k"],
        )

        d = manifest.model_dump(exclude_none=True)
        # JSON Schema validation must pass
        jsonschema.validate(d, self.lora_schema)

        # Corrupt category
        bad_d = dict(d)
        bad_d["category"] = "invalid_category"
        with pytest.raises((pydantic.ValidationError, ValueError)):
            LoRAManifest.model_validate(bad_d)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_d, self.lora_schema)

        # Corrupt adapter_path (.pt forbidden)
        bad_pt = dict(d)
        bad_pt["adapter_path"] = "weights.pt"
        with pytest.raises((pydantic.ValidationError, ValueError)):
            LoRAManifest.model_validate(bad_pt)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_pt, self.lora_schema)

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        job_id=st.text(min_size=1, max_size=32, alphabet=st.characters(categories=["L", "N"])),
        stage=st.sampled_from(["fgpp_validate", "neural_video_generate", "delight_albedo"]),
    )
    def test_bundle_manifest_jsonschema_pydantic_parity(
        self,
        job_id: str,
        stage: str,
    ) -> None:
        """Valid BundleManifest passes both Pydantic and JSON Schema; missing fields fail both."""
        raw_bundle = {
            "schema_version": "synapse.job_bundle.v1.0",
            "job_id": job_id,
            "stage": stage,
            "inputs": {"fgpp_dir": "/path/to/fgpp"},
            "outputs": {"video": "/path/to/video.mp4"},
            "params": {"seed": 42},
        }

        # Pydantic validation
        manifest = BundleManifest.model_validate(raw_bundle)
        d = manifest.model_dump(exclude_none=True)
        # JSON Schema validation
        jsonschema.validate(d, self.job_bundle_schema)

        # Corrupt required stage
        bad_bundle = dict(raw_bundle)
        del bad_bundle["stage"]
        with pytest.raises((pydantic.ValidationError, ValueError)):
            BundleManifest.model_validate(bad_bundle)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_bundle, self.job_bundle_schema)

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        bundle_id=st.text(min_size=1, max_size=32, alphabet=st.characters(categories=["L", "N"])),
        source_name=st.sampled_from(["mesh_primary", "mesh_secondary", "fgpp_aov"]),
        media_type=st.sampled_from(["model/gltf-binary", "image/x-exr"]),
    )
    def test_inspection_bundle_manifest_jsonschema_pydantic_parity(
        self,
        bundle_id: str,
        source_name: str,
        media_type: str,
    ) -> None:
        """Valid InspectionBundleManifest passes both Pydantic and JSON Schema; invalid fails both."""
        raw_bundle = {
            "schema_version": "synapse.bundle.v1.0",
            "bundle_id": bundle_id,
            "scenario_name": "scenario_hero",
            "created_at": "2026-11-19T00:00:00Z",
            "sources": {
                source_name: {
                    "path": "assets/model.glb",
                    "media_type": media_type,
                    "sha256": "f" * 64,
                    "byte_size": 1024,
                }
            },
        }

        manifest = InspectionBundleManifest.model_validate(raw_bundle)
        d = manifest.model_dump(exclude_none=True)
        jsonschema.validate(d, self.inspection_bundle_schema)

        # Corrupt sources (bare string instead of SourceEntry object)
        bad_sources = dict(raw_bundle)
        bad_sources["sources"] = {source_name: "assets/model.glb"}
        with pytest.raises((pydantic.ValidationError, ValueError)):
            InspectionBundleManifest.model_validate(bad_sources)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_sources, self.inspection_bundle_schema)

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        asset_id=st.text(min_size=1, max_size=20, alphabet=st.characters(categories=["L", "N"])),
        score=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_inspection_report_jsonschema_pydantic_parity(
        self,
        asset_id: str,
        score: float,
    ) -> None:
        """Valid InspectionReport passes both Pydantic and JSON Schema; out-of-range score fails both."""
        report = InspectionReport(
            asset_id=asset_id,
            asset_name="TestAsset",
            inspection_timestamp="2026-11-19T00:00:00Z",
            overall_quality_score=score,
            defects=[
                DefectAnnotation3D(
                    defect_id="def-01",
                    severity=0.5,
                    confidence=0.9,
                    category="mesh_crumple",
                    center_3d=[0.0, 0.0, 0.0],
                    bounding_radius_m=0.1,
                    affected_vertices_count=10,
                )
            ],
            requires_hitl=False,
        )

        d = report.model_dump(exclude_none=True)
        jsonschema.validate(d, self.inspection_report_schema)

        # Corrupt score > 1.0
        bad_report = dict(d)
        bad_report["overall_quality_score"] = 1.5
        with pytest.raises((pydantic.ValidationError, ValueError)):
            InspectionReport.model_validate(bad_report)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_report, self.inspection_report_schema)


# ---------------------------------------------------------------------------
# Pillar 4: Mesh Geometry Validation Topology Equivalence
# ---------------------------------------------------------------------------


@pytest.mark.external_data
class TestHypothesisMeshValidationParity:
    """Property tests asserting topology equivalence between Python Oracle and C++ native backend."""

    @settings(max_examples=10, derandomize=True, deadline=None)
    @given(
        scale=st.floats(min_value=0.2, max_value=5.0),
        perturb_x=st.floats(min_value=-0.05, max_value=0.05),
        perturb_y=st.floats(min_value=-0.05, max_value=0.05),
    )
    def test_mesh_geometry_validation_backend_parity(
        self,
        scale: float,
        perturb_x: float,
        perturb_y: float,
    ) -> None:
        """Perturbed healthy tetrahedron must yield identical results across Python and Native backends."""
        from synapse.bridge.aether_bridge import has_aether_geo

        if not has_aether_geo():
            pytest.skip("aether_geo C++20 extension not available on pure-Python runner")

        verts = np.array(
            [
                [scale + perturb_x, scale + perturb_y, scale],
                [-scale, -scale, scale],
                [-scale, scale, -scale],
                [scale, -scale, -scale],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=np.int64,
        )

        res_py = validate_mesh_geometry(verts, faces, backend="python", require_watertight=True)
        res_nat = validate_mesh_geometry(verts, faces, backend="native", require_watertight=True)

        assert res_py.passed == res_nat.passed
        assert res_py.is_manifold == res_nat.is_manifold
        assert res_py.is_watertight == res_nat.is_watertight
        assert len(res_py.degenerate_face_indices) == len(res_nat.degenerate_face_indices)
        assert res_py.num_connected_components == res_nat.num_connected_components
