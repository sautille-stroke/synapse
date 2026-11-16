/// @file test_geometry.cc
/// @brief GoogleTest C++20 unit tests for synapse::geometry.

#include "synapse/geometry/mesh_checks.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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

TEST(GeometryTest, HealthyTetrahedron) {
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

  EXPECT_TRUE(checkManifold(verts, faces, 3));
  EXPECT_TRUE(checkWatertight(verts, faces, 3));
  EXPECT_TRUE(checkDegenerateFaces(verts, faces, 3).empty());
  EXPECT_TRUE(checkSelfIntersections(verts, faces, 3).empty());

  const auto comps = findConnectedComponents(faces, 3);
  ASSERT_EQ(comps.size(), 1u);
  EXPECT_EQ(comps[0].size(), 4u);

  const auto res = validateMeshGeometry(verts, faces, 3, true);
  EXPECT_TRUE(res.passed);
}

TEST(GeometryTest, CoplanarTriangles) {
  // Disjoint coplanar triangles on z = 0 must NOT be flagged.
  const std::vector<Vec3> disjointVerts = {
      Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
      Vec3(2.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0), Vec3(2.0, 1.0, 0.0),
  };
  const std::vector<int64_t> twoFaces = {0, 1, 2, 3, 4, 5};
  EXPECT_TRUE(checkSelfIntersections(disjointVerts, twoFaces, 3).empty());

  // Overlapping coplanar triangles on z = 0 MUST be flagged.
  const std::vector<Vec3> overlapVerts = {
      Vec3(0.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0),
      Vec3(0.5, 0.5, 0.0), Vec3(2.5, 0.5, 0.0), Vec3(0.5, 2.5, 0.0),
  };
  const auto pairs = checkSelfIntersections(overlapVerts, twoFaces, 3);
  ASSERT_EQ(pairs.size(), 1u);
  EXPECT_EQ(pairs[0].first, 0);
  EXPECT_EQ(pairs[0].second, 1);
}

TEST(GeometryTest, BowtieAndEdgeComponents) {
  // Bowtie shares only vertex 2 -> NOT manifold, and 2 edge-connected components.
  const std::vector<Vec3> verts = {
      Vec3(0.0, 0.0, 0.0),
      Vec3(1.0, 0.0, 0.0),
      Vec3(0.5, 0.5, 0.0),
      Vec3(0.0, 1.0, 0.0),
      Vec3(1.0, 1.0, 0.0),
  };
  const std::vector<int64_t> faces = {0, 1, 2, 2, 3, 4};
  EXPECT_FALSE(checkManifold(verts, faces, 3));
  EXPECT_FALSE(checkWatertight(verts, faces, 3));
  // Default num_vertices = 0 must not crash!
  const auto comps = findConnectedComponents(faces, 3);
  EXPECT_EQ(comps.size(), 2u);
}

TEST(GeometryTest, InvalidAndEmptyInputs) {
  const std::vector<Vec3> emptyV;
  const std::vector<int64_t> emptyF;
  EXPECT_FALSE(checkManifold(emptyV, emptyF, 3));
  EXPECT_FALSE(checkWatertight(emptyV, emptyF, 3));
  EXPECT_FALSE(validateMeshGeometry(emptyV, emptyF, 3).passed);

  // Out-of-bounds / negative face indices must fail safely without UB.
  const std::vector<Vec3> verts = {
      Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
  };
  const std::vector<int64_t> oobFaces = {0, -1, 99};
  EXPECT_FALSE(checkManifold(verts, oobFaces, 3));
  EXPECT_EQ(checkDegenerateFaces(verts, oobFaces, 3).size(), 1u);
  EXPECT_TRUE(checkSelfIntersections(verts, oobFaces, 3).empty());

  // Repeated vertex in face must fail manifold.
  const std::vector<int64_t> dupFace = {0, 0, 1};
  EXPECT_FALSE(checkManifold(verts, dupFace, 3));

  // NaN vertex must fail manifold.
  const double nanVal = std::numeric_limits<double>::quiet_NaN();
  const std::vector<Vec3> nanVerts = {
      Vec3(nanVal, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0),
  };
  const std::vector<int64_t> oneFace = {0, 1, 2};
  EXPECT_FALSE(checkManifold(nanVerts, oneFace, 3));
}

}  // namespace
