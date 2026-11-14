#pragma once
/// @file types.h
/// @brief Core geometric types for the synapse native geometry kernel.
///
/// Vec3 is a type alias for aether::DVec3 (double-precision 3D vector from
/// @aether//math).  AABB is a lightweight axis-aligned bounding box built
/// on top of Vec3.  GeometricValidationResult is the aggregate output of
/// the full Tier-0 mesh validation pipeline.

#include <aether/math/dvec.h>  // aether::DVec3
#include <aether/geo/mesh_checks.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace synapse::geometry {

// ---------------------------------------------------------------------------
// Vec3 — alias for aether's double-precision 3D vector
// ---------------------------------------------------------------------------

/// Use aether's double-precision 3D vector throughout the geometry module.
/// All existing code that references `synapse::geometry::Vec3` continues to
/// work unchanged — aether::DVec3 exposes the same .x/.y/.z members, the
/// same arithmetic operators, and the same .dot()/.cross()/.length()/.length_sq()
/// member functions.
using Vec3 = aether::DVec3;

// ---------------------------------------------------------------------------
// AABB — axis-aligned bounding box
// ---------------------------------------------------------------------------

struct AABB {
  Vec3 lo, hi;

  /// Default-construct to an invalid (inverted) box so that the first
  /// expand() call initialises both corners correctly.
  AABB()
      : lo(std::numeric_limits<double>::max(),
           std::numeric_limits<double>::max(),
           std::numeric_limits<double>::max()),
        hi(std::numeric_limits<double>::lowest(),
           std::numeric_limits<double>::lowest(),
           std::numeric_limits<double>::lowest()) {}

  /// Grow the box to contain point @p p.
  void expand(const Vec3& p) {
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x);
    hi.y = std::max(hi.y, p.y);
    hi.z = std::max(hi.z, p.z);
  }

  /// Grow the box to contain another AABB.
  void expand(const AABB& other) {
    expand(other.lo);
    expand(other.hi);
  }

  /// Test whether this box overlaps @p other (inclusive boundaries + epsilon).
  [[nodiscard]] bool overlaps(const AABB& other, double eps = 1e-9) const {
    return lo.x <= other.hi.x + eps && hi.x + eps >= other.lo.x &&
           lo.y <= other.hi.y + eps && hi.y + eps >= other.lo.y &&
           lo.z <= other.hi.z + eps && hi.z + eps >= other.lo.z;
  }
};

// ---------------------------------------------------------------------------
// GeometricValidationResult — alias for aether::geo::MeshValidationResult
// ---------------------------------------------------------------------------

using GeometricValidationResult = aether::geo::MeshValidationResult;

}  // namespace synapse::geometry
