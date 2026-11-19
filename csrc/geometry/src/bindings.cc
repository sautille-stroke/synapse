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

#include <aether/geo/marching_cubes.h>
#include <aether/geo/subdivision.h>
#ifndef SYNAPSE_NO_XATLAS
#include <aether/geo/uv_unwrapper.h>
#endif
#include <aether/spatial/kd_tree.h>

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace aether::core {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool reportAssertionFailure(const char*, const char*, const char*, uint32_t, const char*) {
  return false;
}
}  // namespace aether::core

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

// ---------------------------------------------------------------------------
// Marching Cubes
// ---------------------------------------------------------------------------
py::dict pyMarchingCubes(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& field,
    float isoLevel,
    float cellSize,
    py::sequence origin) {
  const auto buf = field.request();
  if (buf.ndim != 3) {
    throw std::invalid_argument("field must be a 3D float32 array");
  }
  const auto dim_z = static_cast<uint32_t>(buf.shape[0]);
  const auto dim_y = static_cast<uint32_t>(buf.shape[1]);
  const auto dim_x = static_cast<uint32_t>(buf.shape[2]);
  if (dim_z < 2 || dim_y < 2 || dim_x < 2) {
    throw std::invalid_argument("All dimensions of field must be at least 2");
  }
  const float* data = static_cast<const float*>(buf.ptr);

  aether::Vec3 orig(0.0f, 0.0f, 0.0f);
  if (py::len(origin) >= 3) {
    orig = aether::Vec3(origin[0].cast<float>(), origin[1].cast<float>(), origin[2].cast<float>());
  }

  aether::MarchingCubesResult res;
  {
    py::gil_scoped_release release;
    res = aether::MarchingCubes(data, dim_x, dim_y, dim_z, isoLevel, cellSize, orig);
  }

  const size_t numVerts = res.vertices.size();
  const size_t numTris = res.indices.size() / 3;

  py::array_t<double> outVerts({static_cast<py::ssize_t>(numVerts), py::ssize_t(3)});
  py::array_t<int64_t> outFaces({static_cast<py::ssize_t>(numTris), py::ssize_t(3)});
  py::array_t<float> outNormals({static_cast<py::ssize_t>(numVerts), py::ssize_t(3)});

  if (numVerts > 0) {
    auto vBuf = outVerts.request();
    auto* vPtr = static_cast<double*>(vBuf.ptr);
    auto nBuf = outNormals.request();
    auto* nPtr = static_cast<float*>(nBuf.ptr);
    for (size_t i = 0; i < numVerts; ++i) {
      vPtr[i * 3 + 0] = static_cast<double>(res.vertices[i].x);
      vPtr[i * 3 + 1] = static_cast<double>(res.vertices[i].y);
      vPtr[i * 3 + 2] = static_cast<double>(res.vertices[i].z);
      nPtr[i * 3 + 0] = res.normals[i].x;
      nPtr[i * 3 + 1] = res.normals[i].y;
      nPtr[i * 3 + 2] = res.normals[i].z;
    }
  }
  if (numTris > 0) {
    auto fBuf = outFaces.request();
    auto* fPtr = static_cast<int64_t*>(fBuf.ptr);
    for (size_t i = 0; i < res.indices.size(); ++i) {
      fPtr[i] = static_cast<int64_t>(res.indices[i]);
    }
  }

  py::dict d;
  d["vertices"] = outVerts;
  d["faces"] = outFaces;
  d["normals"] = outNormals;
  return d;
}

// ---------------------------------------------------------------------------
// Catmull-Clark Subdivision
// ---------------------------------------------------------------------------
py::dict pyCatmullClarkSubdivide(
    const DoubleArray& vertices,
    const Int64Array& faces,
    int vertsPerFace,
    int levels) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, vertsPerFace);

  aether::geo::SubdMesh mesh;
  mesh.positions.reserve(verts.size());
  for (const auto& v : verts) {
    mesh.positions.push_back(aether::Vec3(static_cast<float>(v.x),
                                          static_cast<float>(v.y),
                                          static_cast<float>(v.z)));
  }

  const int vpf = parsed.vertsPerFace;
  const size_t numFaces = parsed.span.size() / static_cast<size_t>(vpf);
  mesh.faces.reserve(numFaces);
  for (size_t fi = 0; fi < numFaces; ++fi) {
    aether::geo::SubdFace f;
    f.verts.reserve(static_cast<size_t>(vpf));
    for (int k = 0; k < vpf; ++k) {
      f.verts.push_back(static_cast<int>(parsed.span[fi * static_cast<size_t>(vpf) + static_cast<size_t>(k)]));
    }
    mesh.faces.push_back(std::move(f));
  }

  aether::geo::SubdMesh result;
  {
    py::gil_scoped_release release;
    result = mesh.subdivide(levels > 0 ? levels : 1);
  }

  const size_t outNumVerts = result.positions.size();
  py::array_t<double> outVerts({static_cast<py::ssize_t>(outNumVerts), py::ssize_t(3)});
  py::array_t<float> outNormals({static_cast<py::ssize_t>(outNumVerts), py::ssize_t(3)});
  {
    auto vBuf = outVerts.request();
    auto* vPtr = static_cast<double*>(vBuf.ptr);
    auto nBuf = outNormals.request();
    auto* nPtr = static_cast<float*>(nBuf.ptr);
    for (size_t i = 0; i < outNumVerts; ++i) {
      vPtr[i * 3 + 0] = static_cast<double>(result.positions[i].x);
      vPtr[i * 3 + 1] = static_cast<double>(result.positions[i].y);
      vPtr[i * 3 + 2] = static_cast<double>(result.positions[i].z);
      if (i < result.normals.size()) {
        nPtr[i * 3 + 0] = result.normals[i].x;
        nPtr[i * 3 + 1] = result.normals[i].y;
        nPtr[i * 3 + 2] = result.normals[i].z;
      } else {
        nPtr[i * 3 + 0] = 0.0f;
        nPtr[i * 3 + 1] = 1.0f;
        nPtr[i * 3 + 2] = 0.0f;
      }
    }
  }

  const size_t outNumFaces = result.faces.size();
  py::array_t<int64_t> outFaces({static_cast<py::ssize_t>(outNumFaces), py::ssize_t(4)});
  {
    auto fBuf = outFaces.request();
    auto* fPtr = static_cast<int64_t*>(fBuf.ptr);
    for (size_t fi = 0; fi < outNumFaces; ++fi) {
      const auto& face = result.faces[fi];
      for (size_t k = 0; k < 4; ++k) {
        fPtr[fi * 4 + k] = (k < face.verts.size()) ? static_cast<int64_t>(face.verts[k]) : 0;
      }
    }
  }

  py::dict d;
  d["vertices"] = outVerts;
  d["faces"] = outFaces;
  d["normals"] = outNormals;
  return d;
}

// ---------------------------------------------------------------------------
// UV Unwrapping (xatlas)
// ---------------------------------------------------------------------------
#ifdef SYNAPSE_NO_XATLAS
py::dict pyUnwrapUvAtlas(
    const DoubleArray& /* vertices */,
    const Int64Array& /* faces */,
    uint32_t /* maxChartSize */,
    uint32_t /* padding */,
    float /* texelsPerUnit */,
    uint32_t /* resolution */,
    bool /* bilinear */,
    bool /* rotateCharts */) {
  throw std::runtime_error(
      "UV unwrapping (xatlas) is not available in direct-compiled builds. "
      "Use Bazel build: python scripts/build_native.py --backend=bazel");
}
#else
py::dict pyUnwrapUvAtlas(
    const DoubleArray& vertices,
    const Int64Array& faces,
    uint32_t maxChartSize,
    uint32_t padding,
    float texelsPerUnit,
    uint32_t resolution,
    bool bilinear,
    bool rotateCharts) {
  const auto verts = verticesFromNumpy(vertices);
  const auto parsed = facesFromNumpy(faces, 3);
  if (parsed.vertsPerFace != 3) {
    throw std::invalid_argument("unwrap_uv_atlas requires triangular faces (verts_per_face=3)");
  }

  std::vector<aether::Vec3> positions;
  positions.reserve(verts.size());
  for (const auto& v : verts) {
    positions.push_back(aether::Vec3(static_cast<float>(v.x),
                                     static_cast<float>(v.y),
                                     static_cast<float>(v.z)));
  }

  std::vector<aether::u32> indices;
  indices.reserve(parsed.span.size());
  for (const auto idx : parsed.span) {
    indices.push_back(static_cast<aether::u32>(idx));
  }

  aether::geo::UVUnwrapOptions opts;
  opts.maxChartSize = maxChartSize;
  opts.padding = padding;
  opts.texelsPerUnit = texelsPerUnit;
  opts.resolution = resolution;
  opts.bilinear = bilinear;
  opts.rotateCharts = rotateCharts;

  const auto res = [&]() {
    py::gil_scoped_release release;
    return aether::geo::MeshUVUnwrapper::unwrap(positions, indices, opts);
  }();

  if (res.isError()) {
    throw std::runtime_error("UV unwrapping failed with error code: " +
                             std::to_string(static_cast<int>(res.error())));
  }

  const auto& unwrapped = res.value();
  const size_t numUvs = unwrapped.uvs.size();
  py::array_t<float> outUvs({static_cast<py::ssize_t>(numUvs), py::ssize_t(2)});
  {
    auto uvBuf = outUvs.request();
    auto* uvPtr = static_cast<float*>(uvBuf.ptr);
    for (size_t i = 0; i < numUvs; ++i) {
      uvPtr[i * 2 + 0] = unwrapped.uvs[i].x;
      uvPtr[i * 2 + 1] = unwrapped.uvs[i].y;
    }
  }

  const size_t numIndices = unwrapped.indices.size();
  py::array_t<int64_t> outIndices({static_cast<py::ssize_t>(numIndices / 3), py::ssize_t(3)});
  {
    auto idxBuf = outIndices.request();
    auto* idxPtr = static_cast<int64_t*>(idxBuf.ptr);
    for (size_t i = 0; i < numIndices; ++i) {
      idxPtr[i] = static_cast<int64_t>(unwrapped.indices[i]);
    }
  }

  const size_t mapSize = unwrapped.originalVertexMap.size();
  py::array_t<int64_t> outMap(static_cast<py::ssize_t>(mapSize));
  {
    auto mapBuf = outMap.request();
    auto* mapPtr = static_cast<int64_t*>(mapBuf.ptr);
    for (size_t i = 0; i < mapSize; ++i) {
      mapPtr[i] = static_cast<int64_t>(unwrapped.originalVertexMap[i]);
    }
  }

  py::dict d;
  d["uvs"] = outUvs;
  d["indices"] = outIndices;
  d["original_vertex_map"] = outMap;
  d["chart_count"] = unwrapped.chartCount;
  d["utilization"] = unwrapped.utilization;
  d["width"] = unwrapped.width;
  d["height"] = unwrapped.height;
  return d;
}
#endif  // SYNAPSE_NO_XATLAS

// ---------------------------------------------------------------------------
// KD-Tree / BVH Closest Points
// ---------------------------------------------------------------------------
py::tuple pyBvhClosestPoints(
    const DoubleArray& targetPoints,
    const DoubleArray& queryPoints) {
  const auto targets = verticesFromNumpy(targetPoints);
  const auto queries = verticesFromNumpy(queryPoints);

  if (targets.empty() || queries.empty()) {
    py::array_t<int64_t> emptyIndices(0);
    py::array_t<double> emptyDistances(0);
    return py::make_tuple(emptyIndices, emptyDistances);
  }

  struct IndexedPoint {
    aether::Vec3 pos;
    int64_t index;
  };

  std::vector<IndexedPoint> points;
  points.reserve(targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    points.push_back({
        aether::Vec3(static_cast<float>(targets[i].x),
                     static_cast<float>(targets[i].y),
                     static_cast<float>(targets[i].z)),
        static_cast<int64_t>(i)
    });
  }

  aether::KdTree<IndexedPoint, 3, std::function<aether::Vec3(const IndexedPoint&)>> tree;
  tree.Build(points, [](const IndexedPoint& p) { return p.pos; });

  const size_t numQueries = queries.size();
  py::array_t<int64_t> outIndices(static_cast<py::ssize_t>(numQueries));
  py::array_t<double> outDists(static_cast<py::ssize_t>(numQueries));

  auto iBuf = outIndices.request();
  auto* iPtr = static_cast<int64_t*>(iBuf.ptr);
  auto dBuf = outDists.request();
  auto* dPtr = static_cast<double*>(dBuf.ptr);

  {
    py::gil_scoped_release release;
    for (size_t q = 0; q < numQueries; ++q) {
      aether::Vec3 qPos(static_cast<float>(queries[q].x),
                        static_cast<float>(queries[q].y),
                        static_cast<float>(queries[q].z));
      const auto* nearest = tree.Nearest(qPos);
      if (nearest) {
        iPtr[q] = nearest->index;
        const auto& t = targets[static_cast<size_t>(nearest->index)];
        const double dx = queries[q].x - t.x;
        const double dy = queries[q].y - t.y;
        const double dz = queries[q].z - t.z;
        dPtr[q] = std::sqrt(dx * dx + dy * dy + dz * dz);
      } else {
        iPtr[q] = 0;
        dPtr[q] = 0.0;
      }
    }
  }

  return py::make_tuple(outIndices, outDists);
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
      - marching cubes isosurface extraction
      - Catmull-Clark subdivision surfaces
      - xatlas UV atlas parameterization
      - KD-Tree / BVH closest point queries

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

  m.def("marching_cubes", &pyMarchingCubes,
        py::arg("field"), py::arg("iso_level") = 0.0f,
        py::arg("cell_size") = 1.0f,
        py::arg("origin") = py::make_tuple(0.0f, 0.0f, 0.0f),
        "Extract isosurface from a 3D scalar field via Marching Cubes.");

  m.def("catmull_clark_subdivide", &pyCatmullClarkSubdivide,
        py::arg("vertices"), py::arg("faces"),
        py::arg("verts_per_face") = 3, py::arg("levels") = 1,
        "Catmull-Clark subdivision surfaces (produces quad faces).");

  m.def("unwrap_uv_atlas", &pyUnwrapUvAtlas,
        py::arg("vertices"), py::arg("faces"),
        py::arg("max_chart_size") = 0, py::arg("padding") = 2,
        py::arg("texels_per_unit") = 0.0f, py::arg("resolution") = 1024,
        py::arg("bilinear") = true, py::arg("rotate_charts") = true,
        "Non-overlapping UV parameterization and atlas packing via xatlas.");

  m.def("bvh_closest_points", &pyBvhClosestPoints,
        py::arg("target_points"), py::arg("query_points"),
        "Find closest points in target_points for each query_point via KD-Tree.");
}
