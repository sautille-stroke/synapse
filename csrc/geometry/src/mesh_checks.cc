/// @file mesh_checks.cc
/// @brief Implementation of the synapse native geometry validation kernels.
///
/// Algorithm summary
/// -----------------
///   checkManifold         — edge-count map + per-vertex umbrella BFS
///   checkWatertight       — manifold + directed half-edge pairing
///   checkDegenerateFaces  — cross-product area + aspect-ratio heuristic
///   checkSelfIntersections— SpatialGrid broad phase + Möller–Trumbore narrow
///   findConnectedComponents — Union-Find with path compression + union by rank
///   validateMeshGeometry  — orchestrator that calls all of the above
///
/// Vec3 is aliased to aether::DVec3 from @aether//math.

#include "synapse/geometry/mesh_checks.h"

#include "synapse/geometry/aabb.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace synapse::geometry {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

/// Pack two vertex indices into a canonical (undirected) edge key.
/// The smaller index always comes first so that (a,b) == (b,a).
struct EdgeKey {
  int64_t v0, v1;
  EdgeKey(int64_t a, int64_t b) : v0(std::min(a, b)), v1(std::max(a, b)) {}
  bool operator==(const EdgeKey& o) const { return v0 == o.v0 && v1 == o.v1; }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey& e) const {
    // Combine two int64 hashes using a simple mixing scheme.
    auto h1 = std::hash<int64_t>{}(e.v0);
    auto h2 = std::hash<int64_t>{}(e.v1);
    return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

/// Directed half-edge key (order matters).
struct HalfEdge {
  int64_t from, to;
  HalfEdge(int64_t f, int64_t t) : from(f), to(t) {}
  bool operator==(const HalfEdge& o) const {
    return from == o.from && to == o.to;
  }
};

struct HalfEdgeHash {
  size_t operator()(const HalfEdge& e) const {
    auto h1 = std::hash<int64_t>{}(e.from);
    auto h2 = std::hash<int64_t>{}(e.to);
    return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};

/// Compute the number of faces from the flat index buffer.
inline int64_t numFaces(std::span<const int64_t> faces, int vpf) {
  return static_cast<int64_t>(faces.size()) / vpf;
}

/// Convenience: get vertex index of the k-th vertex of face f.
inline int64_t faceVertex(std::span<const int64_t> faces, int vpf,
                          int64_t f, int k) {
  return faces[static_cast<size_t>(f * vpf + k)];
}

// -------------------------------------------------------------------------
// Möller–Trumbore ray–triangle intersection
// -------------------------------------------------------------------------

/// Intersect ray(origin, dir) with triangle (v0, v1, v2).
/// Returns true if an intersection exists with t ∈ [0, 1] (segment test).
/// Uses the classic Möller–Trumbore algorithm.
bool rayTriangleIntersect(const Vec3& origin, const Vec3& dir,
                          const Vec3& v0, const Vec3& v1, const Vec3& v2) {
  constexpr double kEps = 1e-12;
  const Vec3 e1 = v1 - v0;
  const Vec3 e2 = v2 - v0;
  const Vec3 h = dir.cross(e2);
  const double a = e1.dot(h);
  if (std::abs(a) < kEps) return false;  // Ray parallel to triangle.

  const double f = 1.0 / a;
  const Vec3 s = origin - v0;
  const double u = f * s.dot(h);
  if (u < 0.0 || u > 1.0) return false;

  const Vec3 q = s.cross(e1);
  const double v = f * dir.dot(q);
  if (v < 0.0 || u + v > 1.0) return false;

  const double t = f * e2.dot(q);
  // We test a segment, so t must be in [0, 1].
  return t >= 0.0 && t <= 1.0;
}

/// Test whether two triangles intersect.
///
/// Strategy: test each edge of triangle A against the plane of triangle B
/// (and vice versa) using Möller–Trumbore.  Two triangles intersect iff
/// any edge of one pierces the interior of the other.
bool trianglesIntersect(const Vec3& a0, const Vec3& a1, const Vec3& a2,
                        const Vec3& b0, const Vec3& b1, const Vec3& b2) {
  // Test edges of A against triangle B.
  if (rayTriangleIntersect(a0, a1 - a0, b0, b1, b2)) return true;
  if (rayTriangleIntersect(a1, a2 - a1, b0, b1, b2)) return true;
  if (rayTriangleIntersect(a2, a0 - a2, b0, b1, b2)) return true;
  // Test edges of B against triangle A.
  if (rayTriangleIntersect(b0, b1 - b0, a0, a1, a2)) return true;
  if (rayTriangleIntersect(b1, b2 - b1, a0, a1, a2)) return true;
  if (rayTriangleIntersect(b2, b0 - b2, a0, a1, a2)) return true;
  return false;
}

/// Check whether two faces share at least one vertex (adjacent faces).
bool sharesVertex(std::span<const int64_t> faces, int vpf,
                  int64_t fi, int64_t fj) {
  for (int a = 0; a < vpf; ++a) {
    const int64_t va = faceVertex(faces, vpf, fi, a);
    for (int b = 0; b < vpf; ++b) {
      if (va == faceVertex(faces, vpf, fj, b)) return true;
    }
  }
  return false;
}

// -------------------------------------------------------------------------
// Union-Find (Disjoint Set Union) with path compression + union by rank
// -------------------------------------------------------------------------

class UnionFind {
 public:
  explicit UnionFind(int64_t n)
      : m_parent(static_cast<size_t>(n)),
        m_rank(static_cast<size_t>(n), 0) {
    std::iota(m_parent.begin(), m_parent.end(), int64_t{0});
  }

  int64_t find(int64_t x) {
    // Path compression.
    while (m_parent[static_cast<size_t>(x)] != x) {
      m_parent[static_cast<size_t>(x)] =
          m_parent[static_cast<size_t>(m_parent[static_cast<size_t>(x)])];
      x = m_parent[static_cast<size_t>(x)];
    }
    return x;
  }

  void unite(int64_t a, int64_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    // Union by rank.
    if (m_rank[static_cast<size_t>(a)] < m_rank[static_cast<size_t>(b)])
      std::swap(a, b);
    m_parent[static_cast<size_t>(b)] = a;
    if (m_rank[static_cast<size_t>(a)] == m_rank[static_cast<size_t>(b)])
      ++m_rank[static_cast<size_t>(a)];
  }

 private:
  std::vector<int64_t> m_parent;
  std::vector<int64_t> m_rank;
};

}  // anonymous namespace

// ===========================================================================
// Public API implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// checkManifold
// ---------------------------------------------------------------------------
bool checkManifold(std::span<const Vec3> /*vertices*/,
                   std::span<const int64_t> faces,
                   int verts_per_face) {
  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf == 0) return true;  // Vacuously true.

  // ------ Step 1: edge-count check (each edge shared by at most 2 faces) ------
  // Also build per-vertex adjacency (face indices) for the umbrella check.
  std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_count;
  std::unordered_map<int64_t, std::vector<int64_t>> vertex_faces;

  for (int64_t f = 0; f < nf; ++f) {
    for (int k = 0; k < verts_per_face; ++k) {
      const int64_t v0 = faceVertex(faces, verts_per_face, f, k);
      const int64_t v1 = faceVertex(faces, verts_per_face, f,
                                    (k + 1) % verts_per_face);
      EdgeKey ek(v0, v1);
      auto& cnt = edge_count[ek];
      ++cnt;
      if (cnt > 2) return false;  // Non-manifold edge.

      vertex_faces[v0].push_back(f);
    }
  }

  // ------ Step 2: vertex umbrella connectivity ------
  // For each vertex, the set of incident faces must form a single connected
  // component when faces are linked via shared edges around that vertex.
  for (const auto& [vid, inc_faces] : vertex_faces) {
    if (inc_faces.size() <= 1) continue;

    // Build a local adjacency among incident faces: two faces are adjacent
    // iff they share an edge that includes this vertex.
    std::unordered_map<int64_t, std::unordered_set<int64_t>> local_adj;
    for (int64_t fi : inc_faces) local_adj[fi];  // ensure entry

    // For each face, find which other incident faces share an edge through
    // this vertex.
    for (size_t i = 0; i < inc_faces.size(); ++i) {
      const int64_t fi = inc_faces[i];
      // Collect edges of fi that include vid.
      std::vector<EdgeKey> fi_edges;
      for (int k = 0; k < verts_per_face; ++k) {
        const int64_t a = faceVertex(faces, verts_per_face, fi, k);
        const int64_t b = faceVertex(faces, verts_per_face, fi,
                                     (k + 1) % verts_per_face);
        if (a == vid || b == vid) fi_edges.emplace_back(a, b);
      }

      for (size_t j = i + 1; j < inc_faces.size(); ++j) {
        const int64_t fj = inc_faces[j];
        for (int k = 0; k < verts_per_face; ++k) {
          const int64_t a = faceVertex(faces, verts_per_face, fj, k);
          const int64_t b = faceVertex(faces, verts_per_face, fj,
                                       (k + 1) % verts_per_face);
          if (a == vid || b == vid) {
            EdgeKey ek(a, b);
            for (const auto& fe : fi_edges) {
              if (fe == ek) {
                local_adj[fi].insert(fj);
                local_adj[fj].insert(fi);
              }
            }
          }
        }
      }
    }

    // BFS to check single component.
    std::unordered_set<int64_t> visited;
    std::queue<int64_t> bfs_queue;
    bfs_queue.push(inc_faces[0]);
    visited.insert(inc_faces[0]);
    while (!bfs_queue.empty()) {
      int64_t cur = bfs_queue.front();
      bfs_queue.pop();
      for (int64_t nb : local_adj[cur]) {
        if (visited.insert(nb).second) {
          bfs_queue.push(nb);
        }
      }
    }
    if (visited.size() != inc_faces.size()) return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// checkWatertight
// ---------------------------------------------------------------------------
bool checkWatertight(std::span<const Vec3> vertices,
                     std::span<const int64_t> faces,
                     int verts_per_face) {
  // A watertight mesh must first be manifold.
  if (!checkManifold(vertices, faces, verts_per_face)) return false;

  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf == 0) return true;

  // Count directed half-edges.  For a watertight mesh, every directed
  // half-edge (a→b) must have a matching opposite half-edge (b→a),
  // meaning every undirected edge has exactly 2 incident faces with
  // opposite winding.
  std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_face_count;
  std::unordered_map<HalfEdge, int, HalfEdgeHash> half_edge_count;

  for (int64_t f = 0; f < nf; ++f) {
    for (int k = 0; k < verts_per_face; ++k) {
      const int64_t v0 = faceVertex(faces, verts_per_face, f, k);
      const int64_t v1 = faceVertex(faces, verts_per_face, f,
                                    (k + 1) % verts_per_face);
      edge_face_count[EdgeKey(v0, v1)]++;
      half_edge_count[HalfEdge(v0, v1)]++;
    }
  }

  // Every edge must be shared by exactly 2 faces.
  for (const auto& [edge, count] : edge_face_count) {
    if (count != 2) return false;
  }

  // Opposite half-edges must pair up: for every (a→b) there must be (b→a).
  for (const auto& [he, count] : half_edge_count) {
    auto it = half_edge_count.find(HalfEdge(he.to, he.from));
    if (it == half_edge_count.end()) return false;
    if (it->second != count) return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// checkDegenerateFaces
// ---------------------------------------------------------------------------
std::vector<int64_t> checkDegenerateFaces(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    double min_area,
    double max_aspect_ratio) {
  std::vector<int64_t> result;
  const int64_t nf = numFaces(faces, verts_per_face);

  for (int64_t f = 0; f < nf; ++f) {
    // Only implemented for triangles; for higher polygons we'd need
    // triangulation — but the spec targets triangle meshes.
    if (verts_per_face < 3) continue;

    const Vec3& v0 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, f, 0))];
    const Vec3& v1 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, f, 1))];
    const Vec3& v2 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, f, 2))];

    const Vec3 e0 = v1 - v0;
    const Vec3 e1 = v2 - v0;
    const Vec3 e2 = v2 - v1;

    // Area = 0.5 * |cross(e0, e1)|.
    const Vec3 cross_vec = e0.cross(e1);
    const double area = 0.5 * cross_vec.length();

    // Check area threshold.
    if (area < min_area) {
      result.push_back(f);
      continue;
    }

    // Aspect ratio heuristic: longest_edge^2 / (2 * area).
    // For an equilateral triangle this ≈ 1.15; for slivers it grows
    // unboundedly.
    const double longest_sq = std::max({e0.length_sq(),
                                        e1.length_sq(),
                                        e2.length_sq()});
    const double aspect = longest_sq / (2.0 * area);
    if (aspect > max_aspect_ratio) {
      result.push_back(f);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// checkSelfIntersections
// ---------------------------------------------------------------------------
std::vector<std::pair<int64_t, int64_t>> checkSelfIntersections(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face) {
  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf <= 1) return {};

  // ------ Build per-triangle AABBs ------
  std::vector<AABB> boxes(static_cast<size_t>(nf));
  for (int64_t f = 0; f < nf; ++f) {
    AABB& box = boxes[static_cast<size_t>(f)];
    for (int k = 0; k < verts_per_face; ++k) {
      box.expand(vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, f, k))]);
    }
  }

  // ------ Broad phase: spatial grid ------
  SpatialGrid grid(boxes);

  // ------ Narrow phase: Möller–Trumbore triangle–triangle ------
  // Use a set to collect unique sorted pairs.
  std::set<std::pair<int64_t, int64_t>> hit_pairs;

  for (int64_t fi = 0; fi < nf; ++fi) {
    const auto candidates = grid.queryOverlapping(boxes[static_cast<size_t>(fi)], fi);
    for (int64_t fj : candidates) {
      // Canonical ordering to avoid duplicate work.
      if (fj <= fi) continue;

      // Skip pairs that are topologically adjacent (share a vertex).
      if (sharesVertex(faces, verts_per_face, fi, fj)) continue;

      // AABB overlap check (the grid gives cell-level candidates, not
      // exact AABB overlaps — double-check).
      if (!boxes[static_cast<size_t>(fi)].overlaps(
              boxes[static_cast<size_t>(fj)])) {
        continue;
      }

      // Narrow-phase intersection test.
      const Vec3& a0 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fi, 0))];
      const Vec3& a1 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fi, 1))];
      const Vec3& a2 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fi, 2))];
      const Vec3& b0 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fj, 0))];
      const Vec3& b1 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fj, 1))];
      const Vec3& b2 = vertices[static_cast<size_t>(faceVertex(faces, verts_per_face, fj, 2))];

      if (trianglesIntersect(a0, a1, a2, b0, b1, b2)) {
        hit_pairs.emplace(fi, fj);
      }
    }
  }

  return {hit_pairs.begin(), hit_pairs.end()};
}

// ---------------------------------------------------------------------------
// findConnectedComponents
// ---------------------------------------------------------------------------
std::vector<std::vector<int64_t>> findConnectedComponents(
    std::span<const int64_t> faces,
    int verts_per_face,
    int64_t num_vertices) {
  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf == 0) return {};

  // Union-Find on vertex indices: faces sharing a vertex are connected.
  UnionFind uf(num_vertices);

  for (int64_t f = 0; f < nf; ++f) {
    const int64_t v0 = faceVertex(faces, verts_per_face, f, 0);
    for (int k = 1; k < verts_per_face; ++k) {
      uf.unite(v0, faceVertex(faces, verts_per_face, f, k));
    }
  }

  // Group faces by component root.
  std::unordered_map<int64_t, std::vector<int64_t>> comp_map;
  for (int64_t f = 0; f < nf; ++f) {
    const int64_t root = uf.find(faceVertex(faces, verts_per_face, f, 0));
    comp_map[root].push_back(f);
  }

  std::vector<std::vector<int64_t>> components;
  components.reserve(comp_map.size());
  for (auto& [root, face_list] : comp_map) {
    components.push_back(std::move(face_list));
  }

  // Sort components by first face index for deterministic output.
  std::sort(components.begin(), components.end(),
            [](const auto& a, const auto& b) { return a.front() < b.front(); });

  return components;
}

// ---------------------------------------------------------------------------
// validateMeshGeometry
// ---------------------------------------------------------------------------
GeometricValidationResult validateMeshGeometry(
    std::span<const Vec3> vertices,
    std::span<const int64_t> faces,
    int verts_per_face,
    bool require_watertight,
    double min_area,
    double max_aspect_ratio,
    double min_face_ratio) {
  GeometricValidationResult result;
  result.num_vertices = static_cast<int64_t>(vertices.size());
  result.num_faces = numFaces(faces, verts_per_face);

  // --- Manifold check ---
  result.is_manifold = checkManifold(vertices, faces, verts_per_face);
  if (!result.is_manifold) {
    result.failure_reasons.emplace_back("Mesh is not 2-manifold.");
  }

  // --- Watertight check ---
  result.is_watertight = checkWatertight(vertices, faces, verts_per_face);
  if (require_watertight && !result.is_watertight) {
    result.failure_reasons.emplace_back("Mesh is not watertight (required).");
  }

  // --- Degenerate faces ---
  result.degenerate_face_indices =
      checkDegenerateFaces(vertices, faces, verts_per_face,
                           min_area, max_aspect_ratio);
  if (!result.degenerate_face_indices.empty()) {
    result.failure_reasons.emplace_back(
        "Mesh contains " +
        std::to_string(result.degenerate_face_indices.size()) +
        " degenerate face(s).");
  }

  // --- Self-intersections ---
  result.self_intersecting_pairs =
      checkSelfIntersections(vertices, faces, verts_per_face);
  if (!result.self_intersecting_pairs.empty()) {
    result.failure_reasons.emplace_back(
        "Mesh contains " +
        std::to_string(result.self_intersecting_pairs.size()) +
        " self-intersecting face pair(s).");
  }

  // --- Connected components & floating islands ---
  auto components = findConnectedComponents(
      faces, verts_per_face, static_cast<int64_t>(vertices.size()));
  result.num_connected_components = static_cast<int64_t>(components.size());

  // Identify floating islands: components with fewer faces than the
  // threshold (min_face_ratio * total faces).
  const auto island_threshold =
      static_cast<int64_t>(std::ceil(min_face_ratio *
                                     static_cast<double>(result.num_faces)));
  for (const auto& comp : components) {
    if (static_cast<int64_t>(comp.size()) < island_threshold) {
      result.floating_island_face_indices.insert(
          result.floating_island_face_indices.end(),
          comp.begin(), comp.end());
    }
  }
  if (!result.floating_island_face_indices.empty()) {
    result.failure_reasons.emplace_back(
        "Mesh contains " +
        std::to_string(result.floating_island_face_indices.size()) +
        " face(s) in floating island component(s).");
  }

  // --- Overall verdict ---
  result.passed = result.failure_reasons.empty();

  return result;
}

}  // namespace synapse::geometry
