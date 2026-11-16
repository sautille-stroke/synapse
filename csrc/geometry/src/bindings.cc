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

#include <limits>
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

  const auto buf = arr.request();
  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::invalid_argument(
        "vertices array must have shape (N, 3), got ndim=" +
        std::to_string(buf.ndim) + ".");
  }
  if (buf.shape[0] == 0) {
    return {};
  }
  if (buf.shape[0] < 0) {
    throw std::invalid_argument("vertices array has negative shape.");
  }

  const auto* ptr = reinterpret_cast<const Vec3*>(buf.ptr);
  return {ptr, static_cast<size_t>(buf.shape[0])};
}

struct ParsedFaces {
  std::span<const int64_t> span;
  int vertsPerFace{3};
};

/// Accept either a 2-D (F, K) integer array or a 1-D flat index buffer.
ParsedFaces facesFromNumpy(const Int64Array& arr, int defaultVpf) {
  const auto buf = arr.request();
  if (buf.ndim == 2) {
    if (buf.shape[1] < 3 || buf.shape[1] > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
          "2-D faces array must have shape (F, K) with 3 <= K <= INT_MAX, got K=" +
          std::to_string(buf.shape[1]) + ".");
    }
    const int inferredVpf = static_cast<int>(buf.shape[1]);
    if (buf.shape[0] < 0) {
      throw std::invalid_argument("faces array has negative shape[0].");
    }
    const auto rows = static_cast<size_t>(buf.shape[0]);
    const auto cols = static_cast<size_t>(buf.shape[1]);
    if (rows > 0 && cols > std::numeric_limits<size_t>::max() / rows) {
      throw std::overflow_error("faces array dimensions overflow size_t.");
    }
    const size_t total = rows * cols;
    if (total == 0) {
      return {{}, inferredVpf};
    }
    const auto* ptr = static_cast<const int64_t*>(buf.ptr);
    return {{ptr, total}, inferredVpf};
  }

  if (buf.ndim == 1) {
    if (defaultVpf < 3) {
      throw std::invalid_argument(
          "verts_per_face must be >= 3 for 1-D faces buffer, got " +
          std::to_string(defaultVpf) + ".");
    }
    if (buf.shape[0] == 0) {
      return {{}, defaultVpf};
    }
    if (buf.shape[0] < 0) {
      throw std::invalid_argument("faces array has negative size.");
    }
    const auto length = static_cast<size_t>(buf.shape[0]);
    if (length % static_cast<size_t>(defaultVpf) != 0) {
      throw std::invalid_argument(
          "1-D faces buffer length (" + std::to_string(buf.shape[0]) +
          ") must be divisible by verts_per_face (" +
          std::to_string(defaultVpf) + ").");
    }
    const auto* ptr = static_cast<const int64_t*>(buf.ptr);
    return {{ptr, length}, defaultVpf};
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
                     int vertsPerFace) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  py::gil_scoped_release release;
  return checkManifold(verts, parsed.span, parsed.vertsPerFace);
}

bool pyCheckWatertight(const DoubleArray& vertices,
                       const Int64Array& faces,
                       int vertsPerFace) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  py::gil_scoped_release release;
  return checkWatertight(verts, parsed.span, parsed.vertsPerFace);
}

py::array_t<int64_t> pyCheckDegenerateFaces(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int vertsPerFace,
    double minArea,
    double maxAspectRatio) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  std::vector<int64_t> vec;
  {
    py::gil_scoped_release release;
    vec = checkDegenerateFaces(verts, parsed.span, parsed.vertsPerFace,
                               minArea, maxAspectRatio);
  }
  if (vec.size() > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max())) {
    throw std::overflow_error("Too many degenerate faces to fit in pybind11 array.");
  }
  py::array_t<int64_t> out(static_cast<py::ssize_t>(vec.size()));
  if (!vec.empty()) {
    auto outBuf = out.request();
    auto* dst = static_cast<int64_t*>(outBuf.ptr);
    std::copy(vec.begin(), vec.end(), dst);
  }
  return out;
}

std::vector<std::pair<int64_t, int64_t>> pyCheckSelfIntersections(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int vertsPerFace) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  py::gil_scoped_release release;
  return checkSelfIntersections(verts, parsed.span, parsed.vertsPerFace);
}

py::list pyFindConnectedComponents(
    const Int64Array& faces,
    int vertsPerFace,
    int64_t numVertices) {
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  std::vector<std::vector<int64_t>> comps;
  {
    py::gil_scoped_release release;
    comps = findConnectedComponents(parsed.span, parsed.vertsPerFace,
                                    numVertices);
  }
  py::list out;
  for (const auto& comp : comps) {
    if (comp.size() > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max())) {
      throw std::overflow_error("Connected component size overflows py::ssize_t.");
    }
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
    int vertsPerFace,
    bool requireWatertight,
    double minArea,
    double maxAspectRatio,
    double minFaceRatio) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);
  GeometricValidationResult r;
  {
    py::gil_scoped_release release;
    r = validateMeshGeometry(verts, parsed.span, parsed.vertsPerFace,
                             requireWatertight, minArea, maxAspectRatio,
                             minFaceRatio);
  }

  py::list siPairs;
  for (const auto& [a, b] : r.self_intersecting_pairs) {
    siPairs.append(py::make_tuple(a, b));
  }

  py::dict d;
  d["is_manifold"] = r.is_manifold;
  d["is_watertight"] = r.is_watertight;
  d["num_vertices"] = r.num_vertices;
  d["num_faces"] = r.num_faces;
  d["degenerate_face_indices"] = r.degenerate_face_indices;
  d["self_intersecting_pairs"] = siPairs;
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
