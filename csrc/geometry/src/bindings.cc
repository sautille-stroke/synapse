/// @file bindings.cc
/// @brief pybind11 bindings for the synapse native geometry validation kernel.
///
/// Exposes the `_c_geometry` Python module with numpy-based entry points.
/// Vertices are passed as (N, 3) float arrays; faces may be passed as either
/// 2-D (F, K) integer arrays or 1-D flat index buffers of length (F * K).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "synapse/geometry/mesh_checks.h"

#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace synapse::geometry {
namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int64Array = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Helpers: numpy array → span conversions
// ---------------------------------------------------------------------------

/// Interpret a C-contiguous (N, 3) float64 numpy array as a span of Vec3.
std::span<const Vec3> verticesFromNumpy(const DoubleArray& arr) {
  static_assert(sizeof(Vec3) == 3 * sizeof(double),
                "Vec3 must be tightly packed (no padding).");

  auto buf = arr.request();
  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::invalid_argument(
        "vertices array must have shape (N, 3), got ndim=" +
        std::to_string(buf.ndim) + ".");
  }
  if (buf.shape[0] == 0) {
    return {};
  }

  const auto* ptr = reinterpret_cast<const Vec3*>(buf.ptr);
  return {ptr, static_cast<size_t>(buf.shape[0])};
}

struct ParsedFaces {
  std::span<const int64_t> span;
  int verts_per_face;
};

/// Accept either a 2-D (F, K) integer array or a 1-D flat index buffer.
ParsedFaces facesFromNumpy(const Int64Array& arr, int default_vpf) {
  auto buf = arr.request();
  if (buf.ndim == 2) {
    const int inferred_vpf = static_cast<int>(buf.shape[1]);
    if (inferred_vpf < 3) {
      throw std::invalid_argument(
          "2-D faces array must have shape (F, K) with K >= 3, got K=" +
          std::to_string(inferred_vpf) + ".");
    }
    const size_t total =
        static_cast<size_t>(buf.shape[0]) * static_cast<size_t>(buf.shape[1]);
    if (total == 0) {
      return {{}, inferred_vpf};
    }
    const auto* ptr = static_cast<const int64_t*>(buf.ptr);
    return {{ptr, total}, inferred_vpf};
  }

  if (buf.ndim == 1) {
    if (default_vpf < 3) {
      throw std::invalid_argument(
          "verts_per_face must be >= 3 for 1-D faces buffer, got " +
          std::to_string(default_vpf) + ".");
    }
    if (buf.shape[0] == 0) {
      return {{}, default_vpf};
    }
    if (buf.shape[0] % default_vpf != 0) {
      throw std::invalid_argument(
          "1-D faces buffer length (" + std::to_string(buf.shape[0]) +
          ") must be divisible by verts_per_face (" +
          std::to_string(default_vpf) + ").");
    }
    const auto* ptr = static_cast<const int64_t*>(buf.ptr);
    return {{ptr, static_cast<size_t>(buf.shape[0])}, default_vpf};
  }

  throw std::invalid_argument(
      "faces array must be 1-D (flat) or 2-D (F, K), got ndim=" +
      std::to_string(buf.ndim) + ".");
}

// ---------------------------------------------------------------------------
// Wrapped Python-facing functions
// ---------------------------------------------------------------------------

bool pyCheckManifold(const DoubleArray& vertices,
                     const Int64Array& faces,
                     int verts_per_face) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  return checkManifold(verticesFromNumpy(vertices),
                       parsed.span,
                       parsed.verts_per_face);
}

bool pyCheckWatertight(const DoubleArray& vertices,
                       const Int64Array& faces,
                       int verts_per_face) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  return checkWatertight(verticesFromNumpy(vertices),
                         parsed.span,
                         parsed.verts_per_face);
}

py::array_t<int64_t> pyCheckDegenerateFaces(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int verts_per_face,
    double min_area,
    double max_aspect_ratio) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  auto vec = checkDegenerateFaces(verticesFromNumpy(vertices),
                                  parsed.span,
                                  parsed.verts_per_face,
                                  min_area,
                                  max_aspect_ratio);
  py::array_t<int64_t> out(static_cast<py::ssize_t>(vec.size()));
  if (!vec.empty()) {
    auto out_buf = out.request();
    auto* dst = static_cast<int64_t*>(out_buf.ptr);
    std::copy(vec.begin(), vec.end(), dst);
  }
  return out;
}

std::vector<std::pair<int64_t, int64_t>> pyCheckSelfIntersections(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int verts_per_face) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  return checkSelfIntersections(verticesFromNumpy(vertices),
                                parsed.span,
                                parsed.verts_per_face);
}

py::list pyFindConnectedComponents(
    const Int64Array& faces,
    int verts_per_face,
    int64_t num_vertices) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  auto comps = findConnectedComponents(parsed.span,
                                       parsed.verts_per_face,
                                       num_vertices);
  py::list out;
  for (const auto& comp : comps) {
    py::array_t<int64_t> arr(static_cast<py::ssize_t>(comp.size()));
    if (!comp.empty()) {
      auto buf = arr.request();
      std::copy(comp.begin(), comp.end(), static_cast<int64_t*>(buf.ptr));
    }
    out.append(std::move(arr));
  }
  return out;
}

py::dict pyValidateMeshGeometry(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int verts_per_face,
    bool require_watertight,
    double min_area,
    double max_aspect_ratio,
    double min_face_ratio) {
  auto parsed = facesFromNumpy(faces, verts_per_face);
  auto r = validateMeshGeometry(verticesFromNumpy(vertices),
                                parsed.span,
                                parsed.verts_per_face,
                                require_watertight,
                                min_area,
                                max_aspect_ratio,
                                min_face_ratio);

  py::list si_pairs;
  for (const auto& [a, b] : r.self_intersecting_pairs) {
    si_pairs.append(py::make_tuple(a, b));
  }

  py::dict d;
  d["is_manifold"] = r.is_manifold;
  d["is_watertight"] = r.is_watertight;
  d["num_vertices"] = r.num_vertices;
  d["num_faces"] = r.num_faces;
  d["degenerate_face_indices"] = r.degenerate_face_indices;
  d["self_intersecting_pairs"] = si_pairs;
  d["num_connected_components"] = r.num_connected_components;
  d["floating_island_face_indices"] = r.floating_island_face_indices;
  d["passed"] = r.passed;
  d["failure_reasons"] = r.failure_reasons;
  return d;
}

}  // anonymous namespace
}  // namespace synapse::geometry

// ===========================================================================
// Module definition
// ===========================================================================

PYBIND11_MODULE(_c_geometry, m) {
  m.doc() = R"pbdoc(
    synapse._c_geometry — native C++20 geometry validation kernel.

    Provides high-performance mesh topology and geometry checks:
      - 2-manifold validation
      - watertight (closed manifold) validation
      - degenerate face detection (area / aspect ratio)
      - self-intersection detection (AABB broad phase + coplanar SAT / Möller–Trumbore)
      - connected component analysis (edge-sharing Union-Find)

    Vertices: float64 array of shape (N, 3).
    Faces:    int64 2-D array of shape (F, K) or 1-D flat buffer of length (F * K).
  )pbdoc";

  using namespace synapse::geometry;

  m.def("check_manifold", &pyCheckManifold,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        "Check if the mesh is a valid 2-manifold.");

  m.def("check_watertight", &pyCheckWatertight,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        "Check if the mesh is watertight (closed 2-manifold).");

  m.def("check_degenerate_faces", &pyCheckDegenerateFaces,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("min_area") = 1e-8, py::arg("max_aspect_ratio") = 100.0,
        "Find degenerate faces (near-zero area or extreme aspect ratio).");

  m.def("check_self_intersections", &pyCheckSelfIntersections,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        "Find self-intersecting face pairs (sorted (i, j) tuples with i < j).");

  m.def("find_connected_components", &pyFindConnectedComponents,
        py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("num_vertices") = 0,
        "Find edge-connected components sorted largest-first.");

  m.def("validate_mesh_geometry", &pyValidateMeshGeometry,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("require_watertight") = false,
        py::arg("min_area") = 1e-8,
        py::arg("max_aspect_ratio") = 100.0,
        py::arg("min_face_ratio") = 0.01,
        "Full Tier-0 geometric mesh validation.");
}
