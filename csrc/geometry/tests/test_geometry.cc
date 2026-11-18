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

TEST(GeometryTest, QuadCubeWatertight) {
  const std::vector<Vec3> verts = {
      Vec3(-1.0, -1.0, -1.0), Vec3(1.0, -1.0, -1.0),
      Vec3(1.0, 1.0, -1.0),   Vec3(-1.0, 1.0, -1.0),
      Vec3(-1.0, -1.0, 1.0),  Vec3(1.0, -1.0, 1.0),
      Vec3(1.0, 1.0, 1.0),    Vec3(-1.0, 1.0, 1.0),
  };
  const std::vector<int64_t> quadFaces = {
      0, 3, 2, 1,  // -Z bottom
      4, 5, 6, 7,  // +Z top
      0, 1, 5, 4,  // -Y front
      2, 3, 7, 6,  // +Y back
      0, 4, 7, 3,  // -X left
      1, 2, 6, 5,  // +X right
  };

  EXPECT_TRUE(checkManifold(verts, quadFaces, 4));
  EXPECT_TRUE(checkWatertight(verts, quadFaces, 4));
  EXPECT_TRUE(checkDegenerateFaces(verts, quadFaces, 4).empty());
  EXPECT_TRUE(checkSelfIntersections(verts, quadFaces, 4).empty());

  const auto comps = findConnectedComponents(quadFaces, 4);
  ASSERT_EQ(comps.size(), 1u);
  EXPECT_EQ(comps[0].size(), 6u);

  const auto res = validateMeshGeometry(verts, quadFaces, 4, true);
  EXPECT_TRUE(res.passed);
}

TEST(GeometryTest, HighValenceStarPole) {
  constexpr int d = 10000;
  std::vector<Vec3> verts;
  verts.reserve(d + 1);
  verts.push_back(Vec3(0.0, 0.0, 1.0));  // Apex (index 0)

  for (int i = 0; i < d; ++i) {
    const double theta = (2.0 * M_PI * i) / d;
    verts.push_back(Vec3(std::cos(theta), std::sin(theta), 0.0));
  }

  std::vector<int64_t> faces;
  faces.reserve(d * 3);
  for (int i = 0; i < d; ++i) {
    const int next_idx = (i + 1) % d;
    faces.push_back(0);
    faces.push_back(1 + i);
    faces.push_back(1 + next_idx);
  }

  // Open cone is 2-manifold (not watertight because boundary rim has valence 1 edges)
  EXPECT_TRUE(checkManifold(verts, faces, 3));
  EXPECT_FALSE(checkWatertight(verts, faces, 3));
}

TEST(GeometryTest, FloatingIslands) {
  // Main tetrahedron (faces 0..3) + tiny island (face 4)
  const std::vector<Vec3> verts = {
      Vec3(1.0, 1.0, 1.0),
      Vec3(-1.0, -1.0, 1.0),
      Vec3(-1.0, 1.0, -1.0),
      Vec3(1.0, -1.0, -1.0),
      // Disjoint triangle far away
      Vec3(10.0, 10.0, 10.0),
      Vec3(11.0, 10.0, 10.0),
      Vec3(10.0, 11.0, 10.0),
  };
  const std::vector<int64_t> faces = {
      0, 1, 2,
      0, 3, 1,
      0, 2, 3,
      1, 3, 2,
      4, 5, 6,
  };

  const auto comps = findConnectedComponents(faces, 3);
  ASSERT_EQ(comps.size(), 2u);
  EXPECT_EQ(comps[0].size(), 4u);
  EXPECT_EQ(comps[1].size(), 1u);

  // min_face_ratio = 0.25 -> island of 1/5 = 0.20 < 0.25 should be flagged as debris
  const auto res = validateMeshGeometry(verts, faces, 3, false, 1e-8, 100.0, 0.25);
  EXPECT_FALSE(res.passed);
  ASSERT_EQ(res.floating_island_face_indices.size(), 1u);
  EXPECT_EQ(res.floating_island_face_indices[0], 4);
}

TEST(GeometryTest, MarchingCubesSphere) {
  constexpr uint32_t N = 16;
  std::vector<float> field(N * N * N);
  const float center = (N - 1) * 0.5f;
  const float radius = 4.0f;

  for (uint32_t z = 0; z < N; ++z) {
    for (uint32_t y = 0; y < N; ++y) {
      for (uint32_t x = 0; x < N; ++x) {
        const float dx = static_cast<float>(x) - center;
        const float dy = static_cast<float>(y) - center;
        const float dz = static_cast<float>(z) - center;
        field[z * N * N + y * N + x] = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
      }
    }
  }

  const auto res = aether::MarchingCubes(field.data(), N, N, N, 0.0f, 1.0f);
  EXPECT_FALSE(res.vertices.empty());
  EXPECT_FALSE(res.indices.empty());
  EXPECT_EQ(res.indices.size() % 3, 0u);
  EXPECT_EQ(res.vertices.size(), res.normals.size());
}

TEST(GeometryTest, CatmullClarkSubdivideCube) {
  aether::geo::SubdMesh mesh;
  const std::vector<aether::Vec3> v = {
      {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f},
      {1.0f, 1.0f, -1.0f},   {-1.0f, 1.0f, -1.0f},
      {-1.0f, -1.0f, 1.0f},  {1.0f, -1.0f, 1.0f},
      {1.0f, 1.0f, 1.0f},    {-1.0f, 1.0f, 1.0f},
  };
  mesh.positions = v;
  mesh.faces = {
      {{0, 3, 2, 1}},
      {{4, 5, 6, 7}},
      {{0, 1, 5, 4}},
      {{2, 3, 7, 6}},
      {{0, 4, 7, 3}},
      {{1, 2, 6, 5}},
  };

  const auto sub = mesh.subdivide(1);
  EXPECT_EQ(sub.faces.size(), 24u);
  EXPECT_EQ(sub.positions.size(), 26u);
  for (const auto& f : sub.faces) {
    EXPECT_EQ(f.verts.size(), 4u);
  }
}

TEST(GeometryTest, UVUnwrapTetrahedron) {
  const std::vector<aether::Vec3> verts = {
      {1.0f, 1.0f, 1.0f},
      {-1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f},
      {1.0f, -1.0f, -1.0f},
  };
  const std::vector<aether::u32> indices = {
      0, 1, 2,
      0, 3, 1,
      0, 2, 3,
      1, 3, 2,
  };

  const auto res = aether::geo::MeshUVUnwrapper::unwrap(verts, indices);
  ASSERT_TRUE(res.isOk());
  const auto& val = res.value();
  EXPECT_FALSE(val.uvs.empty());
  EXPECT_EQ(val.indices.size(), indices.size());
  for (const auto& uv : val.uvs) {
    EXPECT_GE(uv.x, 0.0f);
    EXPECT_LE(uv.x, 1.0f);
    EXPECT_GE(uv.y, 0.0f);
    EXPECT_LE(uv.y, 1.0f);
  }
}

TEST(GeometryTest, KdTreeClosestPoints) {
  struct Pt {
    aether::Vec3 p;
    int id;
  };
  std::vector<Pt> targets = {
      {{0.0f, 0.0f, 0.0f}, 0},
      {{10.0f, 0.0f, 0.0f}, 1},
      {{0.0f, 10.0f, 0.0f}, 2},
  };
  aether::KdTree<Pt, 3, std::function<aether::Vec3(const Pt&)>> tree;
  tree.Build(targets, [](const Pt& p) { return p.p; });

  const auto* n0 = tree.Nearest(aether::Vec3(0.1f, 0.2f, 0.0f));
  ASSERT_NE(n0, nullptr);
  EXPECT_EQ(n0->id, 0);

  const auto* n1 = tree.Nearest(aether::Vec3(9.5f, 0.1f, -0.1f));
  ASSERT_NE(n1, nullptr);
  EXPECT_EQ(n1->id, 1);
}

// ---------------------------------------------------------------------------
// PR-149: Fuzz and Degenerate Geometry / ASan / UBSan Gate
// ---------------------------------------------------------------------------

TEST(GeometryTest, FuzzDegenerateAndCollinearPolygons) {
  const std::vector<Vec3> fuzzVerts = {
      Vec3(0.0, 0.0, 0.0),
      Vec3(1.0, 0.0, 0.0),
      Vec3(2.0, 0.0, 0.0), // collinear with 0 and 1
      Vec3(0.0, 0.0, 0.0), // duplicate of vertex 0
      Vec3(std::numeric_limits<double>::quiet_NaN(), 1.0, 0.0),
      Vec3(std::numeric_limits<double>::infinity(), 0.0, 0.0),
  };

  const std::vector<int64_t> fuzzFaces = {
      0, 1, 2, // collinear zero-area triangle
      0, 0, 0, // identical index triangle
      0, 3, 1, // triangle containing duplicate vertex
      4, 5, 0, // triangle containing NaN and Inf
  };

  const auto degen = checkDegenerateFaces(fuzzVerts, fuzzFaces, 3);
  EXPECT_FALSE(degen.empty());

  const auto selfInter = checkSelfIntersections(fuzzVerts, fuzzFaces, 3);
  (void)selfInter;

  const auto res = validateMeshGeometry(fuzzVerts, fuzzFaces, 3, false);
  EXPECT_FALSE(res.passed);
}

TEST(GeometryTest, FuzzIndexOutOfBoundsAndNegative) {
  const std::vector<Vec3> verts = {
      Vec3(0.0, 0.0, 0.0),
      Vec3(1.0, 0.0, 0.0),
      Vec3(0.0, 1.0, 0.0),
  };

  const std::vector<int64_t> oobFaces = {
      0, 1, 9999, // out-of-bounds index
      -1, 0, 1,   // negative index
  };

  const auto comps = findConnectedComponents(oobFaces, 3);
  (void)comps;
  const auto res = validateMeshGeometry(verts, oobFaces, 3, false);
  EXPECT_FALSE(res.passed);
}

}  // namespace
