/// @file mesh_checks.cc
/// @brief Implementation of the synapse native geometry validation kernels.
///
/// Algorithm summary
/// -----------------
///   checkManifold           — bounds/finite check + edge-count map + umbrella BFS
///   checkWatertight         — manifold + directed half-edge pairing
///   checkDegenerateFaces    — cross-product area + aspect-ratio heuristic
///   checkSelfIntersections  — SpatialGrid broad phase + coplanar 2D SAT /
///                             Möller–Trumbore narrow phase
///   findConnectedComponents — face-level Union-Find over shared undirected edges
///   validateMeshGeometry    — orchestrator that calls all of the above
///
/// Vec3 is aliased to aether::DVec3 from @aether//math.

#include "synapse/geometry/mesh_checks.h"

#include "synapse/geometry/aabb.h"

#include <algorithm>
#include <array>
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

inline bool isFiniteVec3(const Vec3& v) {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline bool hasValidVertices(std::span<const Vec3> vertices) {
  if (vertices.empty()) return false;
  for (const auto& v : vertices) {
    if (!isFiniteVec3(v)) return false;
  }
  return true;
}

inline bool hasValidFaceBuffer(std::span<const int64_t> faces, int vpf) {
  return vpf >= 3 && !faces.empty() &&
         (faces.size() % static_cast<size_t>(vpf) == 0);
}

inline bool faceIndicesInBounds(std::span<const int64_t> faces,
                                int64_t num_verts) {
  for (int64_t idx : faces) {
    if (idx < 0 || idx >= num_verts) return false;
  }
  return true;
}

/// Compute the number of faces from the flat index buffer.
inline int64_t numFaces(std::span<const int64_t> faces, int vpf) {
  if (vpf <= 0) return 0;
  return static_cast<int64_t>(faces.size()) / vpf;
}

/// Convenience: get vertex index of the k-th vertex of face f.
inline int64_t faceVertex(std::span<const int64_t> faces, int vpf,
                          int64_t f, int k) {
  return faces[static_cast<size_t>(f * vpf + k)];
}

// -------------------------------------------------------------------------
// Exact Coplanar + Non-Coplanar Triangle–Triangle Intersection
// -------------------------------------------------------------------------

/// Intersect segment [origin, origin + dir] with triangle (v0, v1, v2)
/// using Möller–Trumbore with boundary tolerance.
bool segmentPiercesTriangle(const Vec3& origin, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
  constexpr double kParallelEps = 1e-12;
  constexpr double kBaryEps = 1e-9;

  const Vec3 e1 = v1 - v0;
  const Vec3 e2 = v2 - v0;
  const Vec3 h = dir.cross(e2);
  const double a = e1.dot(h);
  if (std::abs(a) < kParallelEps) return false;

  const double inv_a = 1.0 / a;
  const Vec3 s = origin - v0;
  const double u = inv_a * s.dot(h);
  if (u < -kBaryEps || u > 1.0 + kBaryEps) return false;

  const Vec3 q = s.cross(e1);
  const double v = inv_a * dir.dot(q);
  if (v < -kBaryEps || u + v > 1.0 + kBaryEps) return false;

  const double t = inv_a * e2.dot(q);
  return t >= -kBaryEps && t <= 1.0 + kBaryEps;
}

/// Test whether two triangles are separated along an in-plane axis.
bool separatedAlongAxis(const Vec3& axis,
                        const std::array<Vec3, 3>& tri_a,
                        const std::array<Vec3, 3>& tri_b,
                        double eps) {
  const double len = axis.length();
  if (len < 1e-12) return false;
  const Vec3 unit = axis * (1.0 / len);

  double min_a = unit.dot(tri_a[0]);
  double max_a = min_a;
  for (int i = 1; i < 3; ++i) {
    const double p = unit.dot(tri_a[static_cast<size_t>(i)]);
    min_a = std::min(min_a, p);
    max_a = std::max(max_a, p);
  }

  double min_b = unit.dot(tri_b[0]);
  double max_b = min_b;
  for (int i = 1; i < 3; ++i) {
    const double p = unit.dot(tri_b[static_cast<size_t>(i)]);
    min_b = std::min(min_b, p);
    max_b = std::max(max_b, p);
  }

  return max_a < min_b - eps || max_b < min_a - eps;
}

/// Test whether two non-adjacent triangles intersect in 3D space.
///
/// Handles:
///   1. Plane-distance early rejection (parallel non-coplanar planes rejected).
///   2. Coplanar triangles via exact 2D Separating Axis Theorem on the 6
///      in-plane edge normals.
///   3. Non-coplanar triangles via bidirectional segment–triangle piercing.
bool trianglesIntersect(const Vec3& a0, const Vec3& a1, const Vec3& a2,
                        const Vec3& b0, const Vec3& b1, const Vec3& b2) {
  constexpr double kAreaEps = 1e-12;
  constexpr double kPlaneEps = 1e-9;

  const Vec3 n_a = (a1 - a0).cross(a2 - a0);
  const double len_a = n_a.length();
  const Vec3 n_b = (b1 - b0).cross(b2 - b0);
  const double len_b = n_b.length();
  if (len_a < kAreaEps || len_b < kAreaEps) return false;

  const Vec3 unit_a = n_a * (1.0 / len_a);
  const Vec3 unit_b = n_b * (1.0 / len_b);

  const std::array<Vec3, 3> tri_a = {a0, a1, a2};
  const std::array<Vec3, 3> tri_b = {b0, b1, b2};

  // Signed distances of B's vertices to A's plane.
  const double db0 = unit_a.dot(b0 - a0);
  const double db1 = unit_a.dot(b1 - a0);
  const double db2 = unit_a.dot(b2 - a0);
  if (std::min({db0, db1, db2}) > kPlaneEps ||
      std::max({db0, db1, db2}) < -kPlaneEps) {
    return false;
  }

  // Signed distances of A's vertices to B's plane.
  const double da0 = unit_b.dot(a0 - b0);
  const double da1 = unit_b.dot(a1 - b0);
  const double da2 = unit_b.dot(a2 - b0);
  if (std::min({da0, da1, da2}) > kPlaneEps ||
      std::max({da0, da1, da2}) < -kPlaneEps) {
    return false;
  }

  // Coplanar case: all vertices of both triangles lie within kPlaneEps of the
  // shared plane. Run 2D SAT on all 6 in-plane edge normals.
  const double max_abs_db =
      std::max({std::abs(db0), std::abs(db1), std::abs(db2)});
  const double max_abs_da =
      std::max({std::abs(da0), std::abs(da1), std::abs(da2)});
  if (max_abs_db <= kPlaneEps && max_abs_da <= kPlaneEps) {
    const std::array<Vec3, 6> edges = {
        a1 - a0, a2 - a1, a0 - a2,
        b1 - b0, b2 - b1, b0 - b2,
    };
    for (const auto& edge : edges) {
      if (separatedAlongAxis(unit_a.cross(edge), tri_a, tri_b, kPlaneEps)) {
        return false;
      }
    }
    return true;
  }

  // Non-coplanar case: test edges of A against B and edges of B against A.
  if (segmentPiercesTriangle(a0, a1 - a0, b0, b1, b2)) return true;
  if (segmentPiercesTriangle(a1, a2 - a1, b0, b1, b2)) return true;
  if (segmentPiercesTriangle(a2, a0 - a2, b0, b1, b2)) return true;
  if (segmentPiercesTriangle(b0, b1 - b0, a0, a1, a2)) return true;
  if (segmentPiercesTriangle(b1, b2 - b1, a0, a1, a2)) return true;
  if (segmentPiercesTriangle(b2, b0 - b2, a0, a1, a2)) return true;
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
bool checkManifold(std::span<const Vec3> vertices,
                   std::span<const int64_t> faces,
                   int verts_per_face) {
  if (!hasValidVertices(vertices) ||
      !hasValidFaceBuffer(faces, verts_per_face) ||
      !faceIndicesInBounds(faces, static_cast<int64_t>(vertices.size()))) {
    return false;
  }

  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf == 0) return false;

  // ------ Step 1: edge-count check & no duplicate vertices within a face ------
  std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_count;
  std::unordered_map<int64_t, std::vector<int64_t>> vertex_faces;

  for (int64_t f = 0; f < nf; ++f) {
    for (int k = 0; k < verts_per_face; ++k) {
      const int64_t v0 = faceVertex(faces, verts_per_face, f, k);
      const int64_t v1 = faceVertex(faces, verts_per_face, f,
                                    (k + 1) % verts_per_face);
      // Reject duplicate / self-pinching vertices within the same polygon.
      for (int j = k + 1; j < verts_per_face; ++j) {
        if (v0 == faceVertex(faces, verts_per_face, f, j)) {
          return false;
        }
      }

      EdgeKey ek(v0, v1);
      auto& cnt = edge_count[ek];
      ++cnt;
      if (cnt > 2) return false;  // Non-manifold edge (>2 incident faces).

      vertex_faces[v0].push_back(f);
    }
  }

  // ------ Step 2: vertex umbrella connectivity ------
  for (const auto& [vid, inc_faces] : vertex_faces) {
    if (inc_faces.size() <= 1) continue;

    std::unordered_map<int64_t, std::unordered_set<int64_t>> local_adj;
    for (int64_t fi : inc_faces) local_adj[fi];

    for (size_t i = 0; i < inc_faces.size(); ++i) {
      const int64_t fi = inc_faces[i];
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
  if (!checkManifold(vertices, faces, verts_per_face)) return false;

  const int64_t nf = numFaces(faces, verts_per_face);

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

  for (const auto& [edge, count] : edge_face_count) {
    if (count != 2) return false;
  }

  for (const auto& [he, count] : half_edge_count) {
    if (count != 1) return false;
    auto it = half_edge_count.find(HalfEdge(he.to, he.from));
    if (it == half_edge_count.end() || it->second != 1) return false;
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
  if (vertices.empty() || !hasValidFaceBuffer(faces, verts_per_face)) {
    return result;
  }

  const auto num_verts = static_cast<int64_t>(vertices.size());
  const int64_t nf = numFaces(faces, verts_per_face);

  for (int64_t f = 0; f < nf; ++f) {
    const int64_t i0 = faceVertex(faces, verts_per_face, f, 0);
    const int64_t i1 = faceVertex(faces, verts_per_face, f, 1);
    const int64_t i2 = faceVertex(faces, verts_per_face, f, 2);
    if (i0 < 0 || i0 >= num_verts || i1 < 0 || i1 >= num_verts ||
        i2 < 0 || i2 >= num_verts) {
      result.push_back(f);
      continue;
    }

    const Vec3& v0 = vertices[static_cast<size_t>(i0)];
    const Vec3& v1 = vertices[static_cast<size_t>(i1)];
    const Vec3& v2 = vertices[static_cast<size_t>(i2)];
    if (!isFiniteVec3(v0) || !isFiniteVec3(v1) || !isFiniteVec3(v2)) {
      result.push_back(f);
      continue;
    }

    const Vec3 e0 = v1 - v0;
    const Vec3 e1 = v2 - v0;
    const Vec3 e2 = v2 - v1;

    const Vec3 cross_vec = e0.cross(e1);
    const double area = 0.5 * cross_vec.length();

    if (!std::isfinite(area) || area < min_area) {
      result.push_back(f);
      continue;
    }

    const double longest_sq = std::max({e0.length_sq(),
                                        e1.length_sq(),
                                        e2.length_sq()});
    const double aspect = longest_sq / (2.0 * area);
    if (!std::isfinite(aspect) || aspect > max_aspect_ratio) {
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
  if (vertices.empty() || !hasValidFaceBuffer(faces, verts_per_face)) {
    return {};
  }

  const auto num_verts = static_cast<int64_t>(vertices.size());
  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf <= 1) return {};

  // ------ Validate per-face indices & build AABBs ------
  std::vector<AABB> boxes(static_cast<size_t>(nf));
  std::vector<bool> valid_face(static_cast<size_t>(nf), true);

  for (int64_t f = 0; f < nf; ++f) {
    AABB& box = boxes[static_cast<size_t>(f)];
    for (int k = 0; k < verts_per_face; ++k) {
      const int64_t idx = faceVertex(faces, verts_per_face, f, k);
      if (idx < 0 || idx >= num_verts ||
          !isFiniteVec3(vertices[static_cast<size_t>(idx)])) {
        valid_face[static_cast<size_t>(f)] = false;
        break;
      }
      box.expand(vertices[static_cast<size_t>(idx)]);
    }
    if (!valid_face[static_cast<size_t>(f)]) {
      box.lo = Vec3(0.0, 0.0, 0.0);
      box.hi = Vec3(0.0, 0.0, 0.0);
    }
  }

  // ------ Broad phase: spatial grid ------
  SpatialGrid grid(boxes);

  // ------ Narrow phase ------
  std::set<std::pair<int64_t, int64_t>> hit_pairs;

  for (int64_t fi = 0; fi < nf; ++fi) {
    if (!valid_face[static_cast<size_t>(fi)]) continue;
    const auto candidates =
        grid.queryOverlapping(boxes[static_cast<size_t>(fi)], fi);
    for (int64_t fj : candidates) {
      if (fj <= fi || !valid_face[static_cast<size_t>(fj)]) continue;

      if (sharesVertex(faces, verts_per_face, fi, fj)) continue;

      if (!boxes[static_cast<size_t>(fi)].overlaps(
              boxes[static_cast<size_t>(fj)])) {
        continue;
      }

      const Vec3& a0 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fi, 0))];
      const Vec3& a1 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fi, 1))];
      const Vec3& a2 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fi, 2))];
      const Vec3& b0 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fj, 0))];
      const Vec3& b1 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fj, 1))];
      const Vec3& b2 = vertices[static_cast<size_t>(
          faceVertex(faces, verts_per_face, fj, 2))];

      // Skip UV / hard-normal split-vertex seam neighbors (1 or 2 coincident
      // 3D corners, while still flagging 3-vertex duplicate z-fighting faces).
      constexpr double kSeamEpsSq = 1e-18;
      int shared_pos = 0;
      const Vec3 tri_a[3] = {a0, a1, a2};
      const Vec3 tri_b[3] = {b0, b1, b2};
      for (int pa = 0; pa < 3; ++pa) {
        for (int pb = 0; pb < 3; ++pb) {
          const Vec3 diff = tri_a[pa] - tri_b[pb];
          if (diff.dot(diff) <= kSeamEpsSq) {
            ++shared_pos;
            break;
          }
        }
      }
      if (shared_pos == 1 || shared_pos == 2) continue;

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
    int64_t /*num_vertices*/) {
  if (!hasValidFaceBuffer(faces, verts_per_face)) return {};

  const int64_t nf = numFaces(faces, verts_per_face);
  if (nf == 0) return {};

  // Union-Find keyed on face indices [0, nf): faces sharing an undirected
  // polygon edge belong to the same connected surface component.
  UnionFind uf(nf);
  std::unordered_map<EdgeKey, int64_t, EdgeKeyHash> edge_to_face;
  edge_to_face.reserve(static_cast<size_t>(nf * verts_per_face));

  for (int64_t f = 0; f < nf; ++f) {
    for (int k = 0; k < verts_per_face; ++k) {
      const int64_t u = faceVertex(faces, verts_per_face, f, k);
      const int64_t w =
          faceVertex(faces, verts_per_face, f, (k + 1) % verts_per_face);
      auto [it, inserted] = edge_to_face.emplace(EdgeKey(u, w), f);
      if (!inserted) {
        uf.unite(f, it->second);
      }
    }
  }

  std::unordered_map<int64_t, std::vector<int64_t>> comp_map;
  for (int64_t f = 0; f < nf; ++f) {
    comp_map[uf.find(f)].push_back(f);
  }

  std::vector<std::vector<int64_t>> components;
  components.reserve(comp_map.size());
  for (auto& [root, face_list] : comp_map) {
    components.push_back(std::move(face_list));
  }

  // Largest component first; deterministic tie-break on smallest face index.
  std::sort(components.begin(), components.end(),
            [](const auto& a, const auto& b) {
              if (a.size() != b.size()) return a.size() > b.size();
              return a.front() < b.front();
            });

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
  result.num_faces =
      hasValidFaceBuffer(faces, verts_per_face) ? numFaces(faces, verts_per_face) : 0;

  if (vertices.empty() || !hasValidFaceBuffer(faces, verts_per_face)) {
    result.is_manifold = false;
    result.is_watertight = false;
    result.passed = false;
    result.failure_reasons.emplace_back("Mesh is not a valid 2-manifold.");
    if (require_watertight) {
      result.failure_reasons.emplace_back(
          "Mesh is not watertight (has boundary edges).");
    }
    return result;
  }

  // --- Manifold check ---
  result.is_manifold = checkManifold(vertices, faces, verts_per_face);
  if (!result.is_manifold) {
    result.failure_reasons.emplace_back("Mesh is not a valid 2-manifold.");
  }

  // --- Watertight check ---
  result.is_watertight = checkWatertight(vertices, faces, verts_per_face);
  if (require_watertight && !result.is_watertight) {
    result.failure_reasons.emplace_back(
        "Mesh is not watertight (has boundary edges).");
  }

  // --- Degenerate faces ---
  result.degenerate_face_indices =
      checkDegenerateFaces(vertices, faces, verts_per_face,
                           min_area, max_aspect_ratio);
  if (!result.degenerate_face_indices.empty()) {
    result.failure_reasons.emplace_back(
        std::to_string(result.degenerate_face_indices.size()) +
        " degenerate face(s) detected.");
  }

  // --- Self-intersections ---
  result.self_intersecting_pairs =
      checkSelfIntersections(vertices, faces, verts_per_face);
  if (!result.self_intersecting_pairs.empty()) {
    result.failure_reasons.emplace_back(
        std::to_string(result.self_intersecting_pairs.size()) +
        " self-intersecting face pair(s) detected.");
  }

  // --- Connected components & floating islands ---
  auto components = findConnectedComponents(faces, verts_per_face);
  result.num_connected_components = static_cast<int64_t>(components.size());

  const double island_threshold =
      min_face_ratio * static_cast<double>(result.num_faces);
  for (const auto& comp : components) {
    if (static_cast<double>(comp.size()) < island_threshold) {
      result.floating_island_face_indices.insert(
          result.floating_island_face_indices.end(),
          comp.begin(), comp.end());
    }
  }
  std::sort(result.floating_island_face_indices.begin(),
            result.floating_island_face_indices.end());

  if (!result.floating_island_face_indices.empty()) {
    result.failure_reasons.emplace_back(
        std::to_string(result.floating_island_face_indices.size()) +
        " face(s) belong to floating islands.");
  }

  result.passed = result.failure_reasons.empty();
  return result;
}

}  // namespace synapse::geometry
