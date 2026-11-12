#pragma once
/// @file types.h
/// @brief Core geometric types for the synapse native geometry kernel.
///
/// Provides lightweight math primitives (Vec3, AABB) and the aggregate
/// GeometricValidationResult structure returned by the full Tier-0
/// mesh validation pipeline.  No external dependencies beyond the
/// C++ standard library — intentionally header-only for the struct
/// definitions so that downstream translation units can include this
/// without additional link-time cost.

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace synapse::geometry {

// ---------------------------------------------------------------------------
// Vec3 — minimal 3-component double-precision vector
// ---------------------------------------------------------------------------

struct Vec3 {
  double x, y, z;

  Vec3() : x(0), y(0), z(0) {}
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }

  double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }

  Vec3 cross(const Vec3& o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }

  double length_sq() const { return dot(*this); }
  double length() const { return std::sqrt(length_sq()); }
};

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

  /// Test whether this box overlaps @p other (inclusive boundaries).
  bool overlaps(const AABB& other) const {
    return lo.x <= other.hi.x && hi.x >= other.lo.x &&
           lo.y <= other.hi.y && hi.y >= other.lo.y &&
           lo.z <= other.hi.z && hi.z >= other.lo.z;
  }
};

// ---------------------------------------------------------------------------
// GeometricValidationResult — aggregate output of validateMeshGeometry()
// ---------------------------------------------------------------------------

struct GeometricValidationResult {
  bool is_manifold = false;
  bool is_watertight = false;

  int64_t num_vertices = 0;
  int64_t num_faces = 0;

  /// Indices of faces whose area is below the minimum threshold or whose
  /// aspect ratio exceeds the maximum allowed value.
  std::vector<int64_t> degenerate_face_indices;

  /// Pairs of face indices whose triangles geometrically intersect.
  std::vector<std::pair<int64_t, int64_t>> self_intersecting_pairs;

  /// Number of topologically disconnected components.
  int64_t num_connected_components = 0;

  /// Face indices belonging to "floating island" components (any component
  /// whose face count is below `min_face_ratio * num_faces`).
  std::vector<int64_t> floating_island_face_indices;

  /// Overall pass/fail verdict.
  bool passed = false;

  /// Human-readable explanations for each failing criterion.
  std::vector<std::string> failure_reasons;
};

}  // namespace synapse::geometry
