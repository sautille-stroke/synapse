#pragma once
/// @file mesh_checks.h
/// @brief Thin forwarding layer — delegates to @aether//geo mesh validation.
///
/// Every function in synapse::geometry is an inline wrapper that forwards to
/// the corresponding aether::geo function.  This preserves full API and ABI
/// compatibility for bindings.cc, test_geometry.cc, and downstream callers.

#include "synapse/geometry/types.h"

#include <aether/geo/mesh_checks.h>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace synapse::geometry {

/// @brief Check whether the mesh is a valid 2-manifold.
[[nodiscard]] inline bool checkManifold(std::span<const Vec3> vertices,
                                        std::span<const int64_t> faces,
                                        int verts_per_face) {
  return aether::geo::CheckManifold(vertices, faces, verts_per_face);
}

/// @brief Check whether the mesh is watertight (closed 2-manifold).
[[nodiscard]] inline bool checkWatertight(std::span<const Vec3> vertices,
                                          std::span<const int64_t> faces,
                                          int verts_per_face) {
  return aether::geo::CheckWatertight(vertices, faces, verts_per_face);
}

/// @brief Find degenerate faces (near-zero area or extreme aspect ratio).
[[nodiscard]] inline std::vector<int64_t> checkDegenerateFaces(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    double min_area = 1e-8,
    double max_aspect_ratio = 100.0) {
  return aether::geo::CheckDegenerateFaces(vertices, faces, verts_per_face,
                                           min_area, max_aspect_ratio);
}

/// @brief Find self-intersecting face pairs.
[[nodiscard]] inline std::vector<std::pair<int64_t, int64_t>>
checkSelfIntersections(std::span<const Vec3> vertices,
                       std::span<const int64_t> faces,
                       int verts_per_face) {
  return aether::geo::CheckSelfIntersections(vertices, faces, verts_per_face);
}

/// @brief Find connected components via Union-Find over shared edges.
[[nodiscard]] inline std::vector<std::vector<int64_t>> findConnectedComponents(
    std::span<const int64_t> faces,
    int verts_per_face,
    int64_t num_vertices = 0) {
  return aether::geo::FindConnectedComponents(faces, verts_per_face,
                                              num_vertices);
}

/// @brief Full Tier-0 geometric validation.
[[nodiscard]] inline GeometricValidationResult validateMeshGeometry(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    bool require_watertight = false,
    double min_area = 1e-8,
    double max_aspect_ratio = 100.0,
    double min_face_ratio = 0.01) {
  return aether::geo::ValidateMeshGeometry(vertices, faces, verts_per_face,
                                           require_watertight, min_area,
                                           max_aspect_ratio, min_face_ratio);
}

}  // namespace synapse::geometry
