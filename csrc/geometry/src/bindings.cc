/// @file bindings.cc
/// @brief pybind11 bindings for the synapse native geometry validation kernel.
///
/// Exposes the `_c_geometry` Python module with numpy-based entry points.
/// Vertices are passed as (N, 3) float64 arrays; faces as flat int64 arrays.
///
/// Main entry point:
///   validate_mesh_geometry(vertices, faces, verts_per_face, ...) -> dict
///
/// Individual check functions are also exposed for fine-grained use.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "synapse/geometry/mesh_checks.h"

#include <span>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace synapse::geometry {
namespace {

// ---------------------------------------------------------------------------
// Helpers: numpy array → span conversions
// ---------------------------------------------------------------------------

/// Interpret a contiguous (N, 3) float64 numpy array as a span of Vec3.
///
/// Vec3 has 3 consecutive doubles with standard layout, so reinterpret_cast
/// is safe as long as sizeof(Vec3) == 3 * sizeof(double) and the array is
/// C-contiguous.
std::span<const Vec3> verticesFromNumpy(const py::array_t<double>& arr) {
  static_assert(sizeof(Vec3) == 3 * sizeof(double),
                "Vec3 must be tightly packed (no padding).");

  auto buf = arr.request();
  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::invalid_argument(
        "vertices array must have shape (N, 3), got (" +
        std::to_string(buf.shape[0]) + ", " +
        std::to_string(buf.ndim > 1 ? buf.shape[1] : 0) + ").");
  }

  // Ensure C-contiguous layout.
  if (buf.strides[0] != 3 * static_cast<py::ssize_t>(sizeof(double)) ||
      buf.strides[1] != static_cast<py::ssize_t>(sizeof(double))) {
    throw std::invalid_argument(
        "vertices array must be C-contiguous (row-major, no padding).");
  }

  const auto* ptr = reinterpret_cast<const Vec3*>(buf.ptr);
  return {ptr, static_cast<size_t>(buf.shape[0])};
}

/// Interpret a flat int64 numpy array as a span of int64_t.
std::span<const int64_t> facesFromNumpy(const py::array_t<int64_t>& arr) {
  auto buf = arr.request();
  if (buf.ndim != 1) {
    throw std::invalid_argument(
        "faces array must be 1-D (flat index buffer), got ndim=" +
        std::to_string(buf.ndim) + ".");
  }
  const auto* ptr = static_cast<const int64_t*>(buf.ptr);
  return {ptr, static_cast<size_t>(buf.shape[0])};
}

// ---------------------------------------------------------------------------
// Wrapped Python-facing functions
// ---------------------------------------------------------------------------

/// Python wrapper for checkManifold.
bool pyCheckManifold(const py::array_t<double>& vertices,
                     const py::array_t<int64_t>& faces,
                     int verts_per_face) {
  return checkManifold(verticesFromNumpy(vertices),
                       facesFromNumpy(faces),
                       verts_per_face);
}

/// Python wrapper for checkWatertight.
bool pyCheckWatertight(const py::array_t<double>& vertices,
                       const py::array_t<int64_t>& faces,
                       int verts_per_face) {
  return checkWatertight(verticesFromNumpy(vertices),
                         facesFromNumpy(faces),
                         verts_per_face);
}

/// Python wrapper for checkDegenerateFaces.
std::vector<int64_t> pyCheckDegenerateFaces(
    const py::array_t<double>& vertices,
    const py::array_t<int64_t>& faces,
    int verts_per_face,
    double min_area,
    double max_aspect_ratio) {
  return checkDegenerateFaces(verticesFromNumpy(vertices),
                              facesFromNumpy(faces),
                              verts_per_face,
                              min_area,
                              max_aspect_ratio);
}

/// Python wrapper for checkSelfIntersections.
std::vector<std::pair<int64_t, int64_t>> pyCheckSelfIntersections(
    const py::array_t<double>& vertices,
    const py::array_t<int64_t>& faces,
    int verts_per_face) {
  return checkSelfIntersections(verticesFromNumpy(vertices),
                                facesFromNumpy(faces),
                                verts_per_face);
}

/// Python wrapper for findConnectedComponents.
std::vector<std::vector<int64_t>> pyFindConnectedComponents(
    const py::array_t<int64_t>& faces,
    int verts_per_face,
    int64_t num_vertices) {
  return findConnectedComponents(facesFromNumpy(faces),
                                 verts_per_face,
                                 num_vertices);
}

/// Python wrapper for validateMeshGeometry — returns a dict matching the
/// GeometricValidationResult fields.
py::dict pyValidateMeshGeometry(
    const py::array_t<double>& vertices,
    const py::array_t<int64_t>& faces,
    int verts_per_face,
    bool require_watertight,
    double min_area,
    double max_aspect_ratio,
    double min_face_ratio) {
  auto r = validateMeshGeometry(verticesFromNumpy(vertices),
                                facesFromNumpy(faces),
                                verts_per_face,
                                require_watertight,
                                min_area,
                                max_aspect_ratio,
                                min_face_ratio);

  // Convert self_intersecting_pairs to a list of (i, j) tuples.
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
    synapse._c_geometry — native C++ geometry validation kernel.

    Provides high-performance mesh topology and geometry checks:
      - 2-manifold validation
      - watertight (closed manifold) validation
      - degenerate face detection (area / aspect ratio)
      - self-intersection detection (AABB broad phase + Möller–Trumbore)
      - connected component analysis (Union-Find)

    All functions accept numpy arrays directly.
    Vertices: float64 array of shape (N, 3).
    Faces:    int64 flat index buffer of length (F * verts_per_face).
  )pbdoc";

  using namespace synapse::geometry;

  // ---- Individual check functions ----

  m.def("check_manifold", &pyCheckManifold,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        R"pbdoc(
          Check if the mesh is a valid 2-manifold.

          Returns True if every edge has at most 2 incident faces and each
          vertex umbrella is topologically connected.
        )pbdoc");

  m.def("check_watertight", &pyCheckWatertight,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        R"pbdoc(
          Check if the mesh is watertight (closed 2-manifold).

          Returns True if the mesh is manifold and every edge has exactly 2
          faces with opposite winding.
        )pbdoc");

  m.def("check_degenerate_faces", &pyCheckDegenerateFaces,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("min_area") = 1e-8, py::arg("max_aspect_ratio") = 100.0,
        R"pbdoc(
          Find degenerate faces (near-zero area or extreme aspect ratio).

          Returns a list of face indices.
        )pbdoc");

  m.def("check_self_intersections", &pyCheckSelfIntersections,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        R"pbdoc(
          Find self-intersecting face pairs.

          Uses AABB broad-phase + Möller–Trumbore narrow-phase.
          Returns a sorted list of (i, j) tuples where i < j.
        )pbdoc");

  m.def("find_connected_components", &pyFindConnectedComponents,
        py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("num_vertices") = 0,
        R"pbdoc(
          Find connected components via Union-Find.

          Returns a list of lists; each inner list contains the face indices
          belonging to one connected component.
        )pbdoc");

  // ---- Full validation entry point ----

  m.def("validate_mesh_geometry", &pyValidateMeshGeometry,
        py::arg("vertices"), py::arg("faces"), py::arg("verts_per_face") = 3,
        py::arg("require_watertight") = false,
        py::arg("min_area") = 1e-8,
        py::arg("max_aspect_ratio") = 100.0,
        py::arg("min_face_ratio") = 0.01,
        R"pbdoc(
          Full Tier-0 geometric mesh validation.

          Runs manifold, watertight, degenerate-face, self-intersection, and
          connected-component checks.  Returns a dict with fields:
            is_manifold, is_watertight, num_vertices, num_faces,
            degenerate_face_indices, self_intersecting_pairs,
            num_connected_components, floating_island_face_indices,
            passed, failure_reasons.
        )pbdoc");
}
