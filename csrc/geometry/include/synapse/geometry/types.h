#pragma once
/// @file types.h
/// @brief Core geometric types for the synapse native geometry kernel.
///
/// Vec3 is a type alias for aether::DVec3 (double-precision 3D vector from
/// @aether//math). GeometricValidationResult is an alias for
/// aether::geo::MeshValidationResult from @aether//geo.

#include <aether/math/dvec.h>
#include <aether/geo/mesh_checks.h>

namespace synapse::geometry {

/// Double-precision 3D vector from @aether//math.
using Vec3 = aether::DVec3;

/// Full mesh validation result aggregate from @aether//geo.
using GeometricValidationResult = aether::geo::MeshValidationResult;

}  // namespace synapse::geometry
