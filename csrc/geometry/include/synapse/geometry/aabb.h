#pragma once
/// @file aabb.h
/// @brief Uniform-grid spatial acceleration for broad-phase triangle overlap
///        queries.
///
/// SpatialGrid partitions world-space into a regular 3-D grid of cubic cells.
/// Each cell stores the indices of triangles whose AABB overlaps that cell.
/// For a given query triangle, we collect all candidate triangles from the
/// cells its AABB touches, deduplicate, and return them.
///
/// Complexity: O(N) build, O(1) expected per-query (for uniformly distributed
/// geometry), degenerating to O(N) in the worst case (all triangles in one
/// cell).

#include "synapse/geometry/types.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace synapse::geometry {

/// Inclusive low/high cell-index range along each axis.
struct CellRange {
  int lo[3];
  int hi[3];
};

/// A flat uniform spatial grid for broad-phase triangle-pair overlap queries.
class SpatialGrid {
 public:
  /// Build the grid from a set of per-triangle AABBs.
  ///
  /// @param boxes  One AABB per triangle (index-aligned with the face array).
  /// @param resolution  Number of cells along each axis.  If <= 0 the
  ///   resolution is chosen automatically as cbrt(N) clamped to [4, 256].
  explicit SpatialGrid(const std::vector<AABB>& boxes, int resolution = -1) {
    if (boxes.empty()) return;

    const auto num_triangles = static_cast<int64_t>(boxes.size());

    // ------- determine grid resolution -------
    if (resolution <= 0) {
      resolution =
          std::clamp(static_cast<int>(std::cbrt(static_cast<double>(num_triangles))),
                     4, 256);
    }
    m_resolution = resolution;

    // ------- compute world AABB -------
    for (const auto& b : boxes) {
      m_world.expand(b);
    }

    // Add a tiny epsilon padding to avoid edge-case float issues.
    constexpr double kPad = 1e-6;
    m_world.lo = m_world.lo - Vec3(kPad, kPad, kPad);
    m_world.hi = m_world.hi + Vec3(kPad, kPad, kPad);

    // Cell size along each axis.
    m_cell_size = Vec3(
        (m_world.hi.x - m_world.lo.x) / m_resolution,
        (m_world.hi.y - m_world.lo.y) / m_resolution,
        (m_world.hi.z - m_world.lo.z) / m_resolution);

    // Guard against degenerate (flat) meshes -- ensure cell_size > 0.
    if (m_cell_size.x <= 0) m_cell_size.x = 1.0;
    if (m_cell_size.y <= 0) m_cell_size.y = 1.0;
    if (m_cell_size.z <= 0) m_cell_size.z = 1.0;

    // ------- insert triangles into cells -------
    const int64_t total_cells =
        static_cast<int64_t>(m_resolution) * m_resolution * m_resolution;
    m_cells.resize(static_cast<size_t>(total_cells));

    for (int64_t i = 0; i < num_triangles; ++i) {
      CellRange cr = cellRange(boxes[static_cast<size_t>(i)]);
      for (int iz = cr.lo[2]; iz <= cr.hi[2]; ++iz) {
        for (int iy = cr.lo[1]; iy <= cr.hi[1]; ++iy) {
          for (int ix = cr.lo[0]; ix <= cr.hi[0]; ++ix) {
            m_cells[static_cast<size_t>(cellIndex(ix, iy, iz))].push_back(i);
          }
        }
      }
    }
  }

  /// Return the indices of all triangles whose AABB overlaps the cell(s)
  /// occupied by triangle @p face_idx.  The query triangle itself is
  /// excluded from the result.
  ///
  /// @param box  AABB of the query triangle.
  /// @param face_idx  Index of the query triangle (excluded from results).
  [[nodiscard]] std::vector<int64_t> queryOverlapping(const AABB& box,
                                                      int64_t face_idx) const {
    std::unordered_set<int64_t> seen;
    CellRange cr = cellRange(box);
    for (int iz = cr.lo[2]; iz <= cr.hi[2]; ++iz) {
      for (int iy = cr.lo[1]; iy <= cr.hi[1]; ++iy) {
        for (int ix = cr.lo[0]; ix <= cr.hi[0]; ++ix) {
          const auto& cell =
              m_cells[static_cast<size_t>(cellIndex(ix, iy, iz))];
          for (int64_t idx : cell) {
            if (idx != face_idx) {
              seen.insert(idx);
            }
          }
        }
      }
    }
    return {seen.begin(), seen.end()};
  }

 private:
  /// Map a world-space coordinate to a grid cell index along one axis.
  [[nodiscard]] int toCell(double val, double origin, double inv_size) const {
    return std::clamp(static_cast<int>((val - origin) * inv_size),
                      0, m_resolution - 1);
  }

  /// Return the inclusive [lo, hi] cell-index range for an AABB.
  [[nodiscard]] CellRange cellRange(const AABB& box) const {
    const double inv_x = 1.0 / m_cell_size.x;
    const double inv_y = 1.0 / m_cell_size.y;
    const double inv_z = 1.0 / m_cell_size.z;
    CellRange cr;
    cr.lo[0] = toCell(box.lo.x, m_world.lo.x, inv_x);
    cr.lo[1] = toCell(box.lo.y, m_world.lo.y, inv_y);
    cr.lo[2] = toCell(box.lo.z, m_world.lo.z, inv_z);
    cr.hi[0] = toCell(box.hi.x, m_world.lo.x, inv_x);
    cr.hi[1] = toCell(box.hi.y, m_world.lo.y, inv_y);
    cr.hi[2] = toCell(box.hi.z, m_world.lo.z, inv_z);
    return cr;
  }

  /// Linearise a 3-D cell coordinate to a flat index.
  [[nodiscard]] int64_t cellIndex(int x, int y, int z) const {
    return static_cast<int64_t>(z) * m_resolution * m_resolution +
           static_cast<int64_t>(y) * m_resolution + x;
  }

  int m_resolution = 0;
  AABB m_world;
  Vec3 m_cell_size;
  std::vector<std::vector<int64_t>> m_cells;
};

}  // namespace synapse::geometry
