/// @file test_geometry.cc
/// @brief Zero-exception C++20 unit tests for synapse::geometry.

#include "synapse/geometry/mesh_checks.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using synapse::geometry::Vec3;
using synapse::geometry::checkDegenerateFaces;
using synapse::geometry::checkManifold;
using synapse::geometry::checkSelfIntersections;
using synapse::geometry::checkWatertight;
using synapse::geometry::findConnectedComponents;
using synapse::geometry::validateMeshGeometry;

int g_failures = 0;

#define CHECK_COND(expr)                                                  \
  do {                                                                    \
    if (!(expr)) {                                                        \
      std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #expr); \
      ++g_failures;                                                       \
    }                                                                     \
  } while (false)

void testHealthyTetrahedron() {
  const std::vector<Vec3> verts = {
      Vec3(1.0, 1.0, 1.0),
      Vec3(-1.0, -1.0, 1.0),
      Vec3(-1.0, 1.0, -1.0),
      Vec3(1.0, -1.0, -1.0),
  };
  const std::vector<int64_t> faces = {
      0, 1, 2,
      0, 3, 1,
      0, 2, 3,
      1, 3, 2,
  };

  CHECK_COND(checkManifold(verts, faces, 3));
  CHECK_COND(checkWatertight(verts, faces, 3));
  CHECK_COND(checkDegenerateFaces(verts, faces, 3).empty());
  CHECK_COND(checkSelfIntersections(verts, faces, 3).empty());

  const auto comps = findConnectedComponents(faces, 3);
  CHECK_COND(comps.size() == 1);
  CHECK_COND(comps[0].size() == 4);

  const auto res = validateMeshGeometry(verts, faces, 3, true);
  CHECK_COND(res.passed);
}

void testCoplanarTriangles() {
  // Disjoint coplanar triangles on z = 0 must NOT be flagged.
  const std::vector<Vec3> disjoint_verts = {
      Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
      Vec3(2.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0), Vec3(2.0, 1.0, 0.0),
  };
  const std::vector<int64_t> two_faces = {0, 1, 2, 3, 4, 5};
  CHECK_COND(checkSelfIntersections(disjoint_verts, two_faces, 3).empty());

  // Overlapping coplanar triangles on z = 0 MUST be flagged.
  const std::vector<Vec3> overlap_verts = {
      Vec3(0.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0),
      Vec3(0.5, 0.5, 0.0), Vec3(2.5, 0.5, 0.0), Vec3(0.5, 2.5, 0.0),
  };
  const auto pairs = checkSelfIntersections(overlap_verts, two_faces, 3);
  CHECK_COND(pairs.size() == 1);
  CHECK_COND(pairs[0].first == 0 && pairs[0].second == 1);
}

void testBowtieAndEdgeComponents() {
  // Bowtie shares only vertex 2 -> NOT manifold, and 2 edge-connected components.
  const std::vector<Vec3> verts = {
      Vec3(0.0, 0.0, 0.0),
      Vec3(1.0, 0.0, 0.0),
      Vec3(0.5, 0.5, 0.0),
      Vec3(0.0, 1.0, 0.0),
      Vec3(1.0, 1.0, 0.0),
  };
  const std::vector<int64_t> faces = {0, 1, 2, 2, 3, 4};
  CHECK_COND(!checkManifold(verts, faces, 3));
  CHECK_COND(!checkWatertight(verts, faces, 3));
  // Default num_vertices = 0 must not crash!
  const auto comps = findConnectedComponents(faces, 3);
  CHECK_COND(comps.size() == 2);
}

void testInvalidAndEmptyInputs() {
  const std::vector<Vec3> empty_v;
  const std::vector<int64_t> empty_f;
  CHECK_COND(!checkManifold(empty_v, empty_f, 3));
  CHECK_COND(!checkWatertight(empty_v, empty_f, 3));
  CHECK_COND(!validateMeshGeometry(empty_v, empty_f, 3).passed);

  // Out-of-bounds / negative face indices must fail safely without UB.
  const std::vector<Vec3> verts = {
      Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
  };
  const std::vector<int64_t> oob_faces = {0, -1, 99};
  CHECK_COND(!checkManifold(verts, oob_faces, 3));
  CHECK_COND(checkDegenerateFaces(verts, oob_faces, 3).size() == 1);
  CHECK_COND(checkSelfIntersections(verts, oob_faces, 3).empty());

  // Repeated vertex in face must fail manifold.
  const std::vector<int64_t> dup_face = {0, 0, 1};
  CHECK_COND(!checkManifold(verts, dup_face, 3));

  // NaN vertex must fail manifold.
  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  const std::vector<Vec3> nan_verts = {
      Vec3(nan_val, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
  };
  const std::vector<int64_t> one_face = {0, 1, 2};
  CHECK_COND(!checkManifold(nan_verts, one_face, 3));
}

}  // namespace

int main() {
  testHealthyTetrahedron();
  testCoplanarTriangles();
  testBowtieAndEdgeComponents();
  testInvalidAndEmptyInputs();
  if (g_failures == 0) {
    std::printf("All C++20 geometry tests passed.\n");
    return 0;
  }
  std::fprintf(stderr, "%d test check(s) failed.\n", g_failures);
  return 1;
}
