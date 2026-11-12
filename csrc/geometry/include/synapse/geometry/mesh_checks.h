#pragma once
/// @file mesh_checks.h
/// @brief Core mesh validation API for the synapse native geometry kernel.
///
/// Every function operates on a flat index buffer where each run of
/// `verts_per_face` consecutive indices describes one polygon.  For triangle
/// meshes, `verts_per_face == 3`.
///
/// All routines are purely CPU, single-threaded.  The only external dependency
/// is @aether//math for the Vec3 (= aether::DVec3) type.

#include "synapse/geometry/types.h"

#include <span>
#include <vector>

namespace synapse::geometry {

/// Check whether the mesh is a valid 2-manifold.
///
/// A triangle mesh is 2-manifold iff:
///   1. Every edge is shared by at most 2 faces.
///   2. The faces incident to each vertex form a single topological "fan"
///      (the vertex umbrella is connected).
///
/// @param vertices       Vertex position array.
/// @param faces          Flat index buffer (length = num_faces * verts_per_face).
/// @param verts_per_face Number of vertices per face (typically 3).
/// @return true if the mesh is 2-manifold.
[[nodiscard]] bool checkManifold(std::span<const Vec3> vertices,
                                 std::span<const int64_t> faces,
                                 int verts_per_face);

/// Check whether the mesh is watertight (closed 2-manifold).
///
/// A mesh is watertight iff it is 2-manifold **and** every edge has exactly
/// 2 incident faces with opposite winding (directed half-edges cancel).
///
/// @param vertices       Vertex position array.
/// @param faces          Flat index buffer.
/// @param verts_per_face Number of vertices per face.
/// @return true if the mesh is watertight.
[[nodiscard]] bool checkWatertight(std::span<const Vec3> vertices,
                                   std::span<const int64_t> faces,
                                   int verts_per_face);

/// Find degenerate faces (near-zero area or extreme aspect ratio).
///
/// For each triangle, area is computed as 0.5 * |cross(e0, e1)|.  The
/// aspect ratio heuristic is `longest_edge^2 / (2 * area)`.
///
/// @param vertices         Vertex position array.
/// @param faces            Flat index buffer.
/// @param verts_per_face   Number of vertices per face.
/// @param min_area         Minimum triangle area (default 1e-8).
/// @param max_aspect_ratio Maximum aspect ratio (default 100).
/// @return Vector of face indices that are degenerate.
[[nodiscard]] std::vector<int64_t> checkDegenerateFaces(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    double min_area = 1e-8,
    double max_aspect_ratio = 100.0);

/// Find self-intersecting face pairs.
///
/// Uses an AABB-based SpatialGrid for broad-phase culling followed by a
/// Möller–Trumbore narrow-phase triangle–triangle intersection test.
/// Adjacent triangles (sharing ≥1 vertex) are skipped.
///
/// @param vertices       Vertex position array.
/// @param faces          Flat index buffer.
/// @param verts_per_face Number of vertices per face.
/// @return Sorted vector of unique (i, j) pairs where i < j.
[[nodiscard]] std::vector<std::pair<int64_t, int64_t>> checkSelfIntersections(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face);

/// Find connected components via Union-Find.
///
/// Two faces are in the same component if they share at least one vertex.
///
/// @param faces          Flat index buffer.
/// @param verts_per_face Number of vertices per face.
/// @param num_vertices   Total number of unique vertices.
/// @return Vector of components; each component is a vector of face indices.
[[nodiscard]] std::vector<std::vector<int64_t>> findConnectedComponents(
    std::span<const int64_t> faces,
    int verts_per_face,
    int64_t num_vertices);

/// Full Tier-0 geometric validation.
///
/// Orchestrates all individual checks and populates a single aggregate
/// result structure.
///
/// @param vertices          Vertex position array.
/// @param faces             Flat index buffer.
/// @param verts_per_face    Number of vertices per face.
/// @param require_watertight  Fail if mesh is not watertight.
/// @param min_area          Passed to checkDegenerateFaces().
/// @param max_aspect_ratio  Passed to checkDegenerateFaces().
/// @param min_face_ratio    Components with fewer than
///                          `min_face_ratio * num_faces` faces are
///                          classified as floating islands.
/// @return Populated GeometricValidationResult.
[[nodiscard]] GeometricValidationResult validateMeshGeometry(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    bool require_watertight = false,
    double min_area = 1e-8,
    double max_aspect_ratio = 100.0,
    double min_face_ratio = 0.01);

}  // namespace synapse::geometry
