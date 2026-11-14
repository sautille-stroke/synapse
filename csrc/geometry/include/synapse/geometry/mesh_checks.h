#pragma once
/// @file mesh_checks.h
/// @brief Thin forwarding layer — delegates to @aether//geo mesh validation.
///
/// Functions are imported into synapse::geometry via using declarations.
/// This prevents C++ ADL (Argument-Dependent Lookup) ambiguity when callers
/// use unadorned names with aether::DVec3 arguments.

#include "synapse/geometry/types.h"

#include <aether/geo/mesh_checks.h>

namespace synapse::geometry {

using aether::geo::CheckManifold;
using aether::geo::checkManifold;

using aether::geo::CheckWatertight;
using aether::geo::checkWatertight;

using aether::geo::CheckDegenerateFaces;
using aether::geo::checkDegenerateFaces;

using aether::geo::CheckSelfIntersections;
using aether::geo::checkSelfIntersections;

using aether::geo::FindConnectedComponents;
using aether::geo::findConnectedComponents;

using aether::geo::ValidateMeshGeometry;
using aether::geo::validateMeshGeometry;

}  // namespace synapse::geometry
