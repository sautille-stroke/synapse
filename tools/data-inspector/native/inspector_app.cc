/// @file inspector_app.cc
/// @brief Implementation of Data Inspector native application.

#include "inspector_app.h"

#include <aether/fs/filesystem.h>
#include <aether/json.h>
#include <aether/pal/console.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef SYNAPSE_HAS_TINYEXR
#include "tinyexr.h"
#endif

namespace synapse::inspector {

namespace {

std::string_view jsonStringOr(const aether::json::JsonValue& val, std::string_view key,
                              std::string_view fallback = "") noexcept {
  if (!val.isObject() || !val.hasKey(key)) return fallback;
  const auto child = val[key];
  if (!child.isString()) return fallback;
  return child.getStringOr(fallback);
}

double jsonNumberOr(const aether::json::JsonValue& val, std::string_view key,
                    double fallback = 0.0) noexcept {
  if (!val.isObject() || !val.hasKey(key)) return fallback;
  const auto child = val[key];
  if (!child.isNumber()) return fallback;
  return child.getRealOr(fallback);
}

bool jsonBoolOr(const aether::json::JsonValue& val, std::string_view key,
                bool fallback = false) noexcept {
  if (!val.isObject() || !val.hasKey(key)) return fallback;
  const auto child = val[key];
  if (!child.isBool()) return fallback;
  return child.getBoolOr(fallback);
}

}  // namespace

struct StdinQueueState {
  std::mutex mtx;
  std::vector<std::string> lines;
  std::atomic<bool> eof{false};
  std::atomic<bool> stop{false};
};

InspectorApp::InspectorApp(InspectorConfig config)
    : m_config(std::move(config)),
      m_stdinState(std::make_shared<StdinQueueState>()) {
  initDefaultMesh();
}

InspectorApp::~InspectorApp() {
  requestQuit();
  if (m_stdinState) {
    m_stdinState->stop.store(true, std::memory_order_release);
  }
  if (m_stdinThread.joinable()) {
    m_stdinThread.join();
  }
  m_shm.close();
}

void InspectorApp::initDefaultMesh() {
  // Canonical 3D cube (8 vertices, 12 triangles)
  m_mesh.vertices = {
      -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
      -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
  };
  m_mesh.faces = {
      0, 2, 1,  0, 3, 2,  // Front (-Z)
      4, 5, 6,  4, 6, 7,  // Back (+Z)
      0, 1, 5,  0, 5, 4,  // Bottom (-Y)
      3, 6, 2,  3, 7, 6,  // Top (+Y)
      0, 4, 7,  0, 7, 3,  // Left (-X)
      1, 2, 6,  1, 6, 5,  // Right (+X)
  };
  m_mesh.faceNormals.clear();
  for (size_t f = 0; f < m_mesh.faces.size(); f += 3) {
    const uint32_t i0 = m_mesh.faces[f + 0];
    const uint32_t i1 = m_mesh.faces[f + 1];
    const uint32_t i2 = m_mesh.faces[f + 2];
    const float v0x = m_mesh.vertices[i0 * 3 + 0];
    const float v0y = m_mesh.vertices[i0 * 3 + 1];
    const float v0z = m_mesh.vertices[i0 * 3 + 2];
    const float e1x = m_mesh.vertices[i1 * 3 + 0] - v0x;
    const float e1y = m_mesh.vertices[i1 * 3 + 1] - v0y;
    const float e1z = m_mesh.vertices[i1 * 3 + 2] - v0z;
    const float e2x = m_mesh.vertices[i2 * 3 + 0] - v0x;
    const float e2y = m_mesh.vertices[i2 * 3 + 1] - v0y;
    const float e2z = m_mesh.vertices[i2 * 3 + 2] - v0z;
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-6f) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    m_mesh.faceNormals.push_back(nx);
    m_mesh.faceNormals.push_back(ny);
    m_mesh.faceNormals.push_back(nz);
  }
  m_hasCustomMesh = false;
}

bool InspectorApp::loadObjMesh(std::string_view objPath) {
  const std::string pathStr(objPath);
  std::ifstream ifs(pathStr);
  if (!ifs.is_open()) return false;

  std::vector<float> loadedVerts;
  std::vector<uint32_t> loadedFaces;
  std::string line;

  while (std::getline(ifs, line)) {
    if (line.size() < 2) continue;
    if (line[0] == 'v' && line[1] == ' ') {
      std::istringstream iss(line.substr(2));
      float x = 0.0f, y = 0.0f, z = 0.0f;
      if (iss >> x >> y >> z) {
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
          loadedVerts.push_back(x);
          loadedVerts.push_back(y);
          loadedVerts.push_back(z);
        } else {
          // Keep non-finite vertex placeholder at (0,0,0) to preserve index stability.
          loadedVerts.push_back(0.0f);
          loadedVerts.push_back(0.0f);
          loadedVerts.push_back(0.0f);
        }
      }
    } else if (line[0] == 'f' && line[1] == ' ') {
      std::istringstream iss(line.substr(2));
      std::vector<uint32_t> faceIndices;
      std::string token;
      const auto vertCount = static_cast<int64_t>(loadedVerts.size() / 3);

      while (iss >> token) {
        char* endPtr = nullptr;
        long idx = std::strtol(token.c_str(), &endPtr, 10);
        if (idx == 0) continue;  // invalid OBJ index

        uint32_t resolved = 0;
        if (idx > 0) {
          resolved = static_cast<uint32_t>(idx - 1);
        } else {
          // Negative relative index: -1 = last vertex, etc.
          int64_t absIdx = vertCount + idx;
          if (absIdx < 0) continue;  // out of range
          resolved = static_cast<uint32_t>(absIdx);
        }

        // Drop out-of-bounds indices
        if (resolved >= static_cast<uint32_t>(vertCount)) continue;

        faceIndices.push_back(resolved);
      }

      // Fan-triangulate: works for triangles, quads, and N-gons
      if (faceIndices.size() >= 3) {
        for (size_t i = 1; i + 1 < faceIndices.size(); ++i) {
          loadedFaces.push_back(faceIndices[0]);
          loadedFaces.push_back(faceIndices[i]);
          loadedFaces.push_back(faceIndices[i + 1]);
        }
      }
    }
  }

  if (loadedVerts.empty() || loadedFaces.empty()) return false;

  // Center and normalize loaded mesh
  float minB[3] = {1e9f, 1e9f, 1e9f};
  float maxB[3] = {-1e9f, -1e9f, -1e9f};
  for (size_t i = 0; i < loadedVerts.size(); i += 3) {
    for (int c = 0; c < 3; ++c) {
      minB[c] = std::min(minB[c], loadedVerts[i + c]);
      maxB[c] = std::max(maxB[c], loadedVerts[i + c]);
    }
  }
  const float center[3] = {
      (minB[0] + maxB[0]) * 0.5f,
      (minB[1] + maxB[1]) * 0.5f,
      (minB[2] + maxB[2]) * 0.5f,
  };
  const float maxDim = std::max({maxB[0] - minB[0], maxB[1] - minB[1], maxB[2] - minB[2]});
  const float scale = maxDim > 1e-6f ? (2.0f / maxDim) : 1.0f;

  for (size_t i = 0; i < loadedVerts.size(); i += 3) {
    loadedVerts[i + 0] = (loadedVerts[i + 0] - center[0]) * scale;
    loadedVerts[i + 1] = (loadedVerts[i + 1] - center[1]) * scale;
    loadedVerts[i + 2] = (loadedVerts[i + 2] - center[2]) * scale;
  }

  m_mesh.vertices = std::move(loadedVerts);
  m_mesh.faces = std::move(loadedFaces);
  m_mesh.faceNormals.clear();

  const auto vertexCount = m_mesh.vertices.size();
  for (size_t f = 0; f < m_mesh.faces.size(); f += 3) {
    const uint32_t i0 = m_mesh.faces[f + 0];
    const uint32_t i1 = m_mesh.faces[f + 1];
    const uint32_t i2 = m_mesh.faces[f + 2];
    if (i0 * 3 + 2 >= vertexCount ||
        i1 * 3 + 2 >= vertexCount ||
        i2 * 3 + 2 >= vertexCount) {
      m_mesh.faceNormals.push_back(0.0f);
      m_mesh.faceNormals.push_back(1.0f);
      m_mesh.faceNormals.push_back(0.0f);
      continue;
    }
    const float v0x = m_mesh.vertices[i0 * 3 + 0];
    const float v0y = m_mesh.vertices[i0 * 3 + 1];
    const float v0z = m_mesh.vertices[i0 * 3 + 2];
    const float e1x = m_mesh.vertices[i1 * 3 + 0] - v0x;
    const float e1y = m_mesh.vertices[i1 * 3 + 1] - v0y;
    const float e1z = m_mesh.vertices[i1 * 3 + 2] - v0z;
    const float e2x = m_mesh.vertices[i2 * 3 + 0] - v0x;
    const float e2y = m_mesh.vertices[i2 * 3 + 1] - v0y;
    const float e2z = m_mesh.vertices[i2 * 3 + 2] - v0z;
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-6f) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    m_mesh.faceNormals.push_back(nx);
    m_mesh.faceNormals.push_back(ny);
    m_mesh.faceNormals.push_back(nz);
  }

  m_hasCustomMesh = true;
  return true;
}

bool InspectorApp::loadGlbMesh(std::string_view glbPath) {
  // --- Overflow safety constants ---
  // Maximum allowed mesh element counts to prevent integer overflow in byte calculations.
  constexpr size_t kMaxMeshVertices = 16u * 1024u * 1024u;  // 16M vertices
  constexpr size_t kMaxMeshIndices = 48u * 1024u * 1024u;   // 48M indices

  const std::string pathStr(glbPath);
  std::ifstream ifs(pathStr, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) return false;

  const std::streamsize fileSize = ifs.tellg();
  if (fileSize < 12) return false;

  // GLB file size limit: reject files exceeding 256 MiB to prevent heap overflow.
  constexpr size_t kMaxGlbSize = 256u * 1024u * 1024u;  // 256 MiB
  if (static_cast<size_t>(fileSize) > kMaxGlbSize) {
    std::cerr << "[inspector] GLB file exceeds 256 MiB limit: " << pathStr << "\n";
    return false;
  }

  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(fileSize));
  if (!ifs.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
    return false;
  }

  const uint8_t* ptr = buffer.data();
  uint32_t magic = 0, version = 0, length = 0;
  std::memcpy(&magic, ptr, 4);
  std::memcpy(&version, ptr + 4, 4);
  std::memcpy(&length, ptr + 8, 4);

  // glTF binary format magic: 0x46546C67 ("glTF")
  if (magic != 0x46546C67u || version != 2u || length > static_cast<uint32_t>(fileSize)) {
    return false;
  }

  std::string_view jsonChunk;
  std::span<const uint8_t> binChunk;

  size_t pos = 12;
  while (pos + 8 <= buffer.size()) {
    uint32_t chunkLength = 0;
    uint32_t chunkType = 0;
    std::memcpy(&chunkLength, ptr + pos, 4);
    std::memcpy(&chunkType, ptr + pos + 4, 4);
    pos += 8;

    if (pos + chunkLength > buffer.size()) break;

    if (chunkType == 0x4E4F534Au && jsonChunk.empty()) {  // "JSON"
      jsonChunk = std::string_view(reinterpret_cast<const char*>(ptr + pos), chunkLength);
    } else if (chunkType == 0x004E4942u && binChunk.empty()) {  // "BIN\0"
      binChunk = std::span<const uint8_t>(ptr + pos, chunkLength);
    }

    pos += (chunkLength + 3u) & ~3u;
  }

  if (jsonChunk.empty()) return false;

  auto docRes = aether::json::JsonDocument::parse(jsonChunk);
  if (docRes.isError()) return false;
  const auto& root = docRes.value().root();
  if (!root.isObject()) return false;

  const auto accessors = root["accessors"];
  const auto bufferViews = root["bufferViews"];
  const auto meshes = root["meshes"];

  if (!accessors.isArray() || !bufferViews.isArray() || !meshes.isArray() || meshes.size() == 0) {
    return false;
  }

  std::vector<float> loadedVerts;
  std::vector<uint32_t> loadedFaces;

  for (size_t m = 0; m < meshes.size(); ++m) {
    const auto mesh = meshes[m];
    if (!mesh.isObject()) continue;
    const auto prims = mesh["primitives"];
    if (!prims.isArray()) continue;

    for (size_t p = 0; p < prims.size(); ++p) {
      const auto prim = prims[p];
      if (!prim.isObject()) continue;

      // Validate primitive mode: only TRIANGLES (4) is supported.
      const int64_t rawMode = prim["mode"].getIntOr(4);
      if (rawMode != 4) continue;

      const auto attrs = prim["attributes"];
      if (!attrs.isObject() || !attrs.hasKey("POSITION")) continue;

      const int64_t rawPosAccIdx = attrs["POSITION"].getIntOr(-1);
      if (rawPosAccIdx < 0) continue;
      const uint32_t posAccIdx = static_cast<uint32_t>(rawPosAccIdx);
      if (posAccIdx >= accessors.size()) continue;

      const auto posAcc = accessors[posAccIdx];
      if (!posAcc.isObject()) continue;

      // Validate accessor type: POSITION must be VEC3.
      const std::string_view posType = posAcc["type"].getStringOr("");
      if (posType != "VEC3") continue;

      // Signed int64_t pre-validation for all accessor numeric fields.
      const int64_t rawPosCount = posAcc["count"].getIntOr(0);
      const int64_t rawPosCompType = posAcc["componentType"].getIntOr(0);
      const int64_t rawPosViewIdx = posAcc["bufferView"].getIntOr(-1);
      const int64_t rawPosAccOffset = posAcc["byteOffset"].getIntOr(0);

      if (rawPosCount <= 0 || rawPosCompType < 0 || rawPosViewIdx < 0 || rawPosAccOffset < 0) continue;

      // Overflow-safe upper bound check: reject before narrowing cast.
      if (static_cast<uint64_t>(rawPosCount) > kMaxMeshVertices) {
        std::cerr << "[inspector] GLB vertex count exceeds 16M limit: " << rawPosCount << "\n";
        continue;
      }

      const uint32_t posCount = static_cast<uint32_t>(rawPosCount);
      const uint32_t posCompType = static_cast<uint32_t>(rawPosCompType);
      const uint32_t posViewIdx = static_cast<uint32_t>(rawPosViewIdx);
      const size_t posAccOffset = static_cast<size_t>(rawPosAccOffset);

      // POSITION must be FLOAT (5126)
      if (posCompType != 5126u || posViewIdx >= bufferViews.size()) continue;

      const auto posView = bufferViews[posViewIdx];
      if (!posView.isObject()) continue;

      const int64_t rawPosViewOffset = posView["byteOffset"].getIntOr(0);
      const int64_t rawPosViewByteLength = posView["byteLength"].getIntOr(0);
      if (rawPosViewOffset < 0 || rawPosViewByteLength < 0) continue;

      const size_t posViewOffset = static_cast<size_t>(rawPosViewOffset);

      // Overflow-safe addition: posViewOffset + posAccOffset
      if (posViewOffset > binChunk.size() || posAccOffset > binChunk.size() - posViewOffset) continue;
      const size_t posByteOffset = posViewOffset + posAccOffset;

      // Compute byte length with size_t to prevent 32-bit overflow
      const size_t posByteLength = static_cast<size_t>(posCount) * 3u * sizeof(float);
      if (posByteOffset > binChunk.size() || posByteLength > binChunk.size() - posByteOffset) continue;

      // Parse byteStride for interleaved vertex buffers (0 means tightly packed).
      const int64_t rawByteStride = posView["byteStride"].getIntOr(0);
      if (rawByteStride < 0) continue;  // Reject negative byteStride
      const size_t byteStride = (rawByteStride > 0) ? static_cast<size_t>(rawByteStride) : (3u * sizeof(float));
      if (byteStride < 3u * sizeof(float)) continue;  // stride too small for VEC3

      const uint32_t baseVertex = static_cast<uint32_t>(loadedVerts.size() / 3);

      // Use std::memcpy instead of reinterpret_cast to avoid alignment UB on ARM64.
      bool vertsOk = true;
      for (size_t vi = 0; vi < posCount; ++vi) {
        // Overflow-safe multiplication: vi * byteStride
        if (byteStride != 0 && vi > (binChunk.size() / byteStride)) {
          vertsOk = false;
          break;
        }
        const size_t strideOffset = vi * byteStride;
        // Overflow-safe addition: posByteOffset + strideOffset
        if (strideOffset > binChunk.size() || posByteOffset > binChunk.size() - strideOffset) {
          vertsOk = false;
          break;
        }
        const size_t srcOffset = posByteOffset + strideOffset;
        if (srcOffset + 3u * sizeof(float) > binChunk.size()) {
          vertsOk = false;
          break;
        }
        float vx, vy, vz;
        std::memcpy(&vx, binChunk.data() + srcOffset + 0, sizeof(float));
        std::memcpy(&vy, binChunk.data() + srcOffset + sizeof(float), sizeof(float));
        std::memcpy(&vz, binChunk.data() + srcOffset + 2 * sizeof(float), sizeof(float));
        if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
          vertsOk = false;
          break;
        }
        loadedVerts.push_back(vx);
        loadedVerts.push_back(vy);
        loadedVerts.push_back(vz);
      }
      if (!vertsOk) {
        // Rollback partial vertex data for this primitive.
        loadedVerts.resize(static_cast<size_t>(baseVertex) * 3u);
        continue;
      }

      // Load indices if present
      if (prim.hasKey("indices")) {
        const int64_t rawIdxAccIdx = prim["indices"].getIntOr(-1);
        if (rawIdxAccIdx < 0) continue;
        const uint32_t idxAccIdx = static_cast<uint32_t>(rawIdxAccIdx);
        if (idxAccIdx < accessors.size()) {
          const auto idxAcc = accessors[idxAccIdx];

          // Signed pre-validation for index accessor fields.
          const int64_t rawIdxCount = idxAcc["count"].getIntOr(0);
          const int64_t rawIdxCompType = idxAcc["componentType"].getIntOr(0);
          const int64_t rawIdxViewIdx = idxAcc["bufferView"].getIntOr(-1);
          const int64_t rawIdxAccOffset = idxAcc["byteOffset"].getIntOr(0);

          if (rawIdxCount <= 0 || rawIdxCompType < 0 || rawIdxViewIdx < 0 || rawIdxAccOffset < 0) continue;

          // Overflow-safe upper bound check for index count.
          if (static_cast<uint64_t>(rawIdxCount) > kMaxMeshIndices) {
            std::cerr << "[inspector] GLB index count exceeds 48M limit: " << rawIdxCount << "\n";
            continue;
          }

          const uint32_t idxCount = static_cast<uint32_t>(rawIdxCount);
          const uint32_t idxCompType = static_cast<uint32_t>(rawIdxCompType);
          const uint32_t idxViewIdx = static_cast<uint32_t>(rawIdxViewIdx);
          const size_t idxAccOffset = static_cast<size_t>(rawIdxAccOffset);

          if (idxViewIdx < bufferViews.size()) {
            const auto idxView = bufferViews[idxViewIdx];
            const int64_t rawIdxViewOffset = idxView["byteOffset"].getIntOr(0);
            if (rawIdxViewOffset < 0) continue;
            const size_t idxViewOffset = static_cast<size_t>(rawIdxViewOffset);

            // Overflow-safe addition: idxViewOffset + idxAccOffset
            if (idxViewOffset > binChunk.size() || idxAccOffset > binChunk.size() - idxViewOffset) continue;
            const size_t idxByteOffset = idxViewOffset + idxAccOffset;

            // Compute total index byte size with size_t to prevent uint32_t wraparound.
            // Cast idxCount to size_t BEFORE multiplying by element size.
            if (idxCompType == 5121u) {
              // UNSIGNED_BYTE — 1 byte per index, no overflow concern
              const size_t totalIdxBytes = static_cast<size_t>(idxCount);
              if (idxByteOffset > binChunk.size() || totalIdxBytes > binChunk.size() - idxByteOffset) continue;
              const uint8_t* idxData = binChunk.data() + idxByteOffset;
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                if (idxData[i] < posCount && idxData[i + 1] < posCount && idxData[i + 2] < posCount) {
                  loadedFaces.push_back(baseVertex + idxData[i]);
                  loadedFaces.push_back(baseVertex + idxData[i + 1]);
                  loadedFaces.push_back(baseVertex + idxData[i + 2]);
                }
              }
            } else if (idxCompType == 5123u) {
              // UNSIGNED_SHORT — cast to size_t BEFORE multiply to prevent uint32_t overflow
              const size_t totalIdxBytes = static_cast<size_t>(idxCount) * sizeof(uint16_t);
              if (idxByteOffset > binChunk.size() || totalIdxBytes > binChunk.size() - idxByteOffset) continue;
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                uint16_t idx0, idx1, idx2;
                std::memcpy(&idx0, binChunk.data() + idxByteOffset + i * 2, sizeof(uint16_t));
                std::memcpy(&idx1, binChunk.data() + idxByteOffset + (i + 1) * 2, sizeof(uint16_t));
                std::memcpy(&idx2, binChunk.data() + idxByteOffset + (i + 2) * 2, sizeof(uint16_t));
                if (idx0 < posCount && idx1 < posCount && idx2 < posCount) {
                  loadedFaces.push_back(baseVertex + idx0);
                  loadedFaces.push_back(baseVertex + idx1);
                  loadedFaces.push_back(baseVertex + idx2);
                }
              }
            } else if (idxCompType == 5125u) {
              // UNSIGNED_INT — cast to size_t BEFORE multiply to prevent uint32_t overflow
              const size_t totalIdxBytes = static_cast<size_t>(idxCount) * sizeof(uint32_t);
              if (idxByteOffset > binChunk.size() || totalIdxBytes > binChunk.size() - idxByteOffset) continue;
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                uint32_t idx0, idx1, idx2;
                std::memcpy(&idx0, binChunk.data() + idxByteOffset + i * 4, sizeof(uint32_t));
                std::memcpy(&idx1, binChunk.data() + idxByteOffset + (i + 1) * 4, sizeof(uint32_t));
                std::memcpy(&idx2, binChunk.data() + idxByteOffset + (i + 2) * 4, sizeof(uint32_t));
                if (idx0 < posCount && idx1 < posCount && idx2 < posCount) {
                  loadedFaces.push_back(baseVertex + idx0);
                  loadedFaces.push_back(baseVertex + idx1);
                  loadedFaces.push_back(baseVertex + idx2);
                }
              }
            }
          }
        }
      } else {
        // No indices -> sequential triangles
        for (uint32_t i = 0; i + 2 < posCount; i += 3) {
          loadedFaces.push_back(baseVertex + i);
          loadedFaces.push_back(baseVertex + i + 1);
          loadedFaces.push_back(baseVertex + i + 2);
        }
      }
    }
  }

  if (loadedVerts.empty() || loadedFaces.empty()) return false;

  // Center and normalize loaded mesh
  float minB[3] = {1e9f, 1e9f, 1e9f};
  float maxB[3] = {-1e9f, -1e9f, -1e9f};
  for (size_t i = 0; i < loadedVerts.size(); i += 3) {
    for (int c = 0; c < 3; ++c) {
      minB[c] = std::min(minB[c], loadedVerts[i + c]);
      maxB[c] = std::max(maxB[c], loadedVerts[i + c]);
    }
  }
  const float center[3] = {
      (minB[0] + maxB[0]) * 0.5f,
      (minB[1] + maxB[1]) * 0.5f,
      (minB[2] + maxB[2]) * 0.5f,
  };
  const float maxDim = std::max({maxB[0] - minB[0], maxB[1] - minB[1], maxB[2] - minB[2]});
  const float scale = maxDim > 1e-6f ? (2.0f / maxDim) : 1.0f;

  for (size_t i = 0; i < loadedVerts.size(); i += 3) {
    loadedVerts[i + 0] = (loadedVerts[i + 0] - center[0]) * scale;
    loadedVerts[i + 1] = (loadedVerts[i + 1] - center[1]) * scale;
    loadedVerts[i + 2] = (loadedVerts[i + 2] - center[2]) * scale;
  }

  m_mesh.vertices = std::move(loadedVerts);
  m_mesh.faces = std::move(loadedFaces);
  m_mesh.faceNormals.clear();

  const auto vertexCount = m_mesh.vertices.size();
  for (size_t f = 0; f < m_mesh.faces.size(); f += 3) {
    const uint32_t i0 = m_mesh.faces[f + 0];
    const uint32_t i1 = m_mesh.faces[f + 1];
    const uint32_t i2 = m_mesh.faces[f + 2];
    if (i0 * 3 + 2 >= vertexCount ||
        i1 * 3 + 2 >= vertexCount ||
        i2 * 3 + 2 >= vertexCount) {
      m_mesh.faceNormals.push_back(0.0f);
      m_mesh.faceNormals.push_back(1.0f);
      m_mesh.faceNormals.push_back(0.0f);
      continue;
    }
    const float v0x = m_mesh.vertices[i0 * 3 + 0];
    const float v0y = m_mesh.vertices[i0 * 3 + 1];
    const float v0z = m_mesh.vertices[i0 * 3 + 2];
    const float e1x = m_mesh.vertices[i1 * 3 + 0] - v0x;
    const float e1y = m_mesh.vertices[i1 * 3 + 1] - v0y;
    const float e1z = m_mesh.vertices[i1 * 3 + 2] - v0z;
    const float e2x = m_mesh.vertices[i2 * 3 + 0] - v0x;
    const float e2y = m_mesh.vertices[i2 * 3 + 1] - v0y;
    const float e2z = m_mesh.vertices[i2 * 3 + 2] - v0z;
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-6f) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    m_mesh.faceNormals.push_back(nx);
    m_mesh.faceNormals.push_back(ny);
    m_mesh.faceNormals.push_back(nz);
  }

  m_hasCustomMesh = true;
  return true;
}

bool InspectorApp::loadExrAov(const std::string& path) {
#ifdef SYNAPSE_HAS_TINYEXR
  float* out = nullptr;
  int width = 0, height = 0;
  const char* err = nullptr;
  int ret = LoadEXR(&out, &width, &height, path.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      std::cerr << "[inspector] EXR load error: " << err << "\n";
      FreeEXRErrorMessage(err);
    }
    return false;
  }

  if (width <= 0 || height <= 0 || !out) {
    if (out) free(out);
    return false;
  }

  const size_t pixelCount = static_cast<size_t>(width) * height;
  m_exrAovData.assign(out, out + pixelCount * 4);  // RGBA float
  m_exrAovWidth = static_cast<uint32_t>(width);
  m_exrAovHeight = static_cast<uint32_t>(height);
  m_hasExrAov = true;
  free(out);

  std::cerr << "[inspector] Loaded EXR AOV: " << path
            << " (" << width << "x" << height << ")\n";
  return true;
#else
  std::cerr << "[inspector] EXR loading not available (tinyexr not linked): "
            << path << "\n";
  (void)path;
  return false;
#endif
}

bool InspectorApp::validateAndLoadBundle(std::string_view bundlePath) {
  if (bundlePath.empty()) return false;

  // 1. Reset to baseline mesh on reload or failure
  initDefaultMesh();

  std::error_code ec;
  std::filesystem::path bundleRoot = std::filesystem::weakly_canonical(bundlePath, ec);
  if (ec || !std::filesystem::exists(bundleRoot) || !std::filesystem::is_directory(bundleRoot)) {
    return false;
  }

  std::filesystem::path manifestPath = bundleRoot / "bundle.json";
  auto textRes = aether::fs::readAllText(manifestPath.string());
  if (textRes.isError()) {
    std::cerr << "[inspector] Bundle manifest not found: " << manifestPath.string() << "\n";
    return false;
  }

  auto docRes = aether::json::JsonDocument::parse(textRes.value());
  if (docRes.isError()) {
    std::cerr << "[inspector] Malformed bundle manifest: " << manifestPath.string() << "\n";
    return false;
  }

  m_config.bundlePath = bundleRoot.string();
  const auto& j = docRes.value().root();
  if (!j.isObject()) return false;

  auto tryLoadFile = [&](std::string_view rawRelPath) {
    if (rawRelPath.empty()) return;
    std::filesystem::path candidatePath(rawRelPath);
    if (candidatePath.is_absolute()) {
      std::cerr << "[inspector] Absolute path rejected in bundle manifest: " << rawRelPath << "\n";
      return;
    }

    std::filesystem::path resolved = std::filesystem::weakly_canonical(bundleRoot / candidatePath, ec);
    if (ec) return;

    // Path traversal check: must resolve inside bundleRoot
    auto rel = std::filesystem::relative(resolved, bundleRoot, ec);
    if (ec || rel.empty() || rel.string().starts_with("..") || rel.is_absolute()) {
      std::cerr << "[inspector] Path traversal rejected: " << rawRelPath << "\n";
      return;
    }

    const std::string ext = resolved.extension().string();
    if (ext == ".glb") {
      if (loadGlbMesh(resolved.string())) {
        std::cerr << "[inspector] Loaded GLB mesh: " << resolved.string() << "\n";
      }
    } else if (ext == ".obj") {
      if (loadObjMesh(resolved.string())) {
        std::cerr << "[inspector] Loaded OBJ mesh: " << resolved.string() << "\n";
      }
    } else if (ext == ".exr") {
      if (loadExrAov(resolved.string())) {
        std::cerr << "[inspector] Loaded EXR AOV: " << resolved.string() << "\n";
      }
    }
  };

  if (j.hasKey("sources") && j["sources"].isObject()) {
    j["sources"].forEachMember([&](std::string_view, aether::json::JsonValue val) {
      if (val.isString()) {
        tryLoadFile(val.getStringOr(""));
      } else if (val.isObject() && val.hasKey("path") && val["path"].isString()) {
        tryLoadFile(val["path"].getStringOr(""));
      } else if (val.isArray()) {
        val.forEachElement([&](size_t, aether::json::JsonValue elem) {
          if (elem.isString()) {
            tryLoadFile(elem.getStringOr(""));
          } else if (elem.isObject() && elem.hasKey("path") && elem["path"].isString()) {
            tryLoadFile(elem["path"].getStringOr(""));
          }
        });
      }
    });
  }

  std::cerr << "[inspector] Loaded bundle: " << m_config.bundlePath
            << " (verts=" << (m_mesh.vertices.size() / 3)
            << ", faces=" << (m_mesh.faces.size() / 3) << ")\n";
  return true;
}

bool InspectorApp::init() {
  if (!m_config.bundlePath.empty()) {
    if (!validateAndLoadBundle(m_config.bundlePath)) {
      return false;
    }
  }

  // Open SHM: file-backed or POSIX shm_open depending on config.
  bool shmOk = false;
  if (m_config.useShmFile && !m_config.shmFilePath.empty()) {
    shmOk = m_shm.openFile(m_config.shmFilePath, m_config.width, m_config.height);
    if (!shmOk) {
      std::cerr << "[inspector] Failed to open file-backed SHM: " << m_config.shmFilePath << "\n";
      return false;
    }
    std::cerr << "[inspector] File-backed SHM opened: " << m_config.shmFilePath << " ("
              << m_config.width << "x" << m_config.height << ", "
              << m_shm.totalSize() << " bytes)\n";
  } else {
    shmOk = m_shm.open(m_config.shmName, m_config.width, m_config.height,
                        !m_config.keepShm);
    if (!shmOk) {
      std::cerr << "[inspector] Failed to open SHM region: " << m_config.shmName << "\n";
      return false;
    }
    std::cerr << "[inspector] SHM opened: " << m_config.shmName << " ("
              << m_config.width << "x" << m_config.height << ", "
              << m_shm.totalSize() << " bytes)\n";
  }
  std::cerr << "[inspector] RHI backend: " << m_config.rhiBackend
            << " (diagnostic SHM rasterizer)\n";
  std::cerr << "[inspector] Headless: "
            << (m_config.headless ? "true" : "false") << "\n";

  // Prevent double thread spawn on repeated init()
  if (m_stdinThread.joinable()) {
    if (m_stdinState) {
      m_stdinState->stop.store(true, std::memory_order_release);
    }
    m_stdinThread.join();
    m_stdinState = std::make_shared<StdinQueueState>();
  }

  auto state = m_stdinState;
  m_stdinThread = std::thread([state]() {
    std::string lineBuf;
    constexpr size_t kMaxLineSize = 64 * 1024;  // 64 KiB max line length
    constexpr size_t kChunkSize = 4096;
    char chunk[kChunkSize];

    while (!state->stop.load(std::memory_order_acquire)) {
      auto result = aether::pal::pollStdinBytes(chunk, kChunkSize);

      if (result.pipeClosed) {
        if (!lineBuf.empty()) {
          if (lineBuf.back() == '\r') {
            lineBuf.pop_back();
          }
          if (!lineBuf.empty()) {
            std::lock_guard<std::mutex> lock(state->mtx);
            state->lines.push_back(std::move(lineBuf));
            lineBuf.clear();
          }
        }
        break;
      }

      if (result.bytesRead == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      if (lineBuf.size() + result.bytesRead > kMaxLineSize) {
        // Discard oversized line to protect against OOM
        lineBuf.clear();
        continue;
      }

      lineBuf.append(chunk, result.bytesRead);
      size_t pos = 0;
      while (true) {
        const auto nl = lineBuf.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string oneLine = lineBuf.substr(pos, nl - pos);
        pos = nl + 1;
        if (!oneLine.empty() && oneLine.back() == '\r') {
          oneLine.pop_back();
        }
        if (oneLine.empty()) continue;
        std::lock_guard<std::mutex> lock(state->mtx);
        state->lines.push_back(std::move(oneLine));
      }
      if (pos > 0) {
        lineBuf.erase(0, pos);
      }
    }
    state->eof.store(true, std::memory_order_release);
  });

  return true;
}

void InspectorApp::run() {
  m_running.store(true, std::memory_order_release);

  using Clock = std::chrono::high_resolution_clock;
  auto lastTime = Clock::now();
  uint64_t fpsFrameCounter = 0;
  uint64_t totalFrames = 0;
  float accumulatedTime = 0.0f;
  float measuredFps = 60.0f;

  std::cerr << "[inspector] Entering frame loop\n";

  while (m_running.load(std::memory_order_acquire)) {
    const auto now = Clock::now();
    const float deltaTime =
        std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    accumulatedTime += deltaTime;
    fpsFrameCounter++;
    if (accumulatedTime >= 0.5f) {
      measuredFps = static_cast<float>(fpsFrameCounter) / accumulatedTime;
      fpsFrameCounter = 0;
      accumulatedTime = 0.0f;
    }

    pollStdinCommands();

    if (!m_running.load(std::memory_order_acquire)) {
      break;
    }

    renderFrame(deltaTime);
    m_shm.publishFrame(measuredFps, deltaTime, static_cast<uint8_t>(m_debugView));

    // Emit structured JSON-RPC frame notification on stdout (pure JSON only)
    aether::json::JsonWriter frameWriter;
    frameWriter.beginObject();
    frameWriter.write("type", "frame_ready");
    frameWriter.endObject();
    std::cout << frameWriter.toString(false) << "\n";
    std::cout.flush();

    totalFrames++;
    if (m_config.maxFrames > 0 && totalFrames >= m_config.maxFrames) {
      std::cerr << "[inspector] Reached maxFrames (" << m_config.maxFrames
                << "), exiting\n";
      requestQuit();
      break;
    }

    if (m_stdinState->eof.load(std::memory_order_acquire) && m_config.maxFrames == 0) {
      bool empty = true;
      {
        std::lock_guard<std::mutex> lock(m_stdinState->mtx);
        empty = m_stdinState->lines.empty();
      }
      if (empty) {
        requestQuit();
        break;
      }
    }

    const auto frameEnd = Clock::now();
    const float elapsed =
        std::chrono::duration<float>(frameEnd - now).count();
    const float sleepTime = (1.0f / 60.0f) - elapsed;
    if (sleepTime > 0.001f) {
      std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
    }
  }

  std::cerr << "[inspector] Frame loop exited\n";
}

void InspectorApp::renderFrame(float /*deltaTime*/) {
  uint8_t* buf = m_shm.frontBuffer();
  if (!buf) return;

  const uint32_t w = m_config.width;
  const uint32_t h = m_config.height;
  const size_t pixelCount = static_cast<size_t>(w) * h;

  // Clear background: dark aesthetic viewport palette (RGBA = 16, 20, 28, 255)
  for (uint32_t i = 0; i < w * h; ++i) {
    buf[i * 4 + 0] = 16;
    buf[i * 4 + 1] = 20;
    buf[i * 4 + 2] = 28;
    buf[i * 4 + 3] = 255;
  }

  if (!m_entityVisible) return;

  // Reuse persistent Z-buffer to eliminate heap reallocation per frame at 60 Hz
  if (m_zBuffer.size() != pixelCount) {
    m_zBuffer.resize(pixelCount);
  }
  std::fill(m_zBuffer.begin(), m_zBuffer.end(), std::numeric_limits<float>::max());

  // View matrix transformation parameters
  const float cy = std::cos(m_orbitYaw);
  const float sy = std::sin(m_orbitYaw);
  const float cp = std::cos(m_orbitPitch);
  const float sp = std::sin(m_orbitPitch);

  const float fovScale = std::min(w, h) * 0.9f;
  const float cx = w * 0.5f;
  const float cy_screen = h * 0.5f;

  // Transform 3D points to screen space
  auto projectPoint = [&](float x, float y, float z, float& outSx, float& outSy, float& outCamZ) -> bool {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;

    // Rotate Y (yaw)
    const float x1 = cy * x + sy * z;
    const float y1 = y;
    const float z1 = -sy * x + cy * z;
    // Rotate X (pitch)
    const float x2 = x1;
    const float y2 = cp * y1 - sp * z1;
    const float z2 = sp * y1 + cp * z1;
    // Pan & orbit distance
    const float xc = x2 + m_panX;
    const float yc = y2 + m_panY;
    const float zc = z2 + m_orbitDistance;

    outCamZ = zc;
    if (zc <= 0.2f || !std::isfinite(zc)) return false;

    outSx = cx + (xc / zc) * fovScale;
    outSy = cy_screen - (yc / zc) * fovScale;
    return std::isfinite(outSx) && std::isfinite(outSy);  // Let the rasterizer AABB clipping handle culling
  };

  // Bresenham's 2D line drawer
  auto drawLine = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int syy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
      if (x0 >= 0 && x0 < static_cast<int>(w) && y0 >= 0 && y0 < static_cast<int>(h)) {
        const size_t off = (static_cast<size_t>(y0) * w + static_cast<size_t>(x0)) * 4u;
        buf[off + 0] = r;
        buf[off + 1] = g;
        buf[off + 2] = b;
        buf[off + 3] = 255;
      }
      if (x0 == x1 && y0 == y1) break;
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += syy; }
    }
  };

  // Draw ground grid
  const float gridY = -1.0f;
  for (int g = -2; g <= 2; ++g) {
    float gx = static_cast<float>(g);
    float p0x, p0y, p0z, p1x, p1y, p1z;
    if (projectPoint(gx, gridY, -2.0f, p0x, p0y, p0z) &&
        projectPoint(gx, gridY,  2.0f, p1x, p1y, p1z)) {
      drawLine(static_cast<int>(p0x), static_cast<int>(p0y),
               static_cast<int>(p1x), static_cast<int>(p1y), 32, 40, 56);
    }
    if (projectPoint(-2.0f, gridY, gx, p0x, p0y, p0z) &&
        projectPoint( 2.0f, gridY, gx, p1x, p1y, p1z)) {
      drawLine(static_cast<int>(p0x), static_cast<int>(p0y),
               static_cast<int>(p1x), static_cast<int>(p1y), 32, 40, 56);
    }
  }

  // Directional light vector (world space)
  const float lx = 0.577f, ly = 0.577f, lz = 0.577f;

  // Render mesh triangles
  const size_t vertexCount = m_mesh.vertices.size();
  const size_t numFaces = m_mesh.faces.size() / 3;
  for (size_t f = 0; f < numFaces; ++f) {
    const uint32_t i0 = m_mesh.faces[f * 3 + 0];
    const uint32_t i1 = m_mesh.faces[f * 3 + 1];
    const uint32_t i2 = m_mesh.faces[f * 3 + 2];

    // Bounds check face indices against vertex array
    if (i0 * 3 + 2 >= vertexCount ||
        i1 * 3 + 2 >= vertexCount ||
        i2 * 3 + 2 >= vertexCount) {
      continue;
    }

    float sx0, sy0, sz0;
    float sx1, sy1, sz1;
    float sx2, sy2, sz2;

    const bool p0_ok = projectPoint(m_mesh.vertices[i0 * 3 + 0],
                                    m_mesh.vertices[i0 * 3 + 1],
                                    m_mesh.vertices[i0 * 3 + 2], sx0, sy0, sz0);
    const bool p1_ok = projectPoint(m_mesh.vertices[i1 * 3 + 0],
                                    m_mesh.vertices[i1 * 3 + 1],
                                    m_mesh.vertices[i1 * 3 + 2], sx1, sy1, sz1);
    const bool p2_ok = projectPoint(m_mesh.vertices[i2 * 3 + 0],
                                    m_mesh.vertices[i2 * 3 + 1],
                                    m_mesh.vertices[i2 * 3 + 2], sx2, sy2, sz2);

    if (!p0_ok || !p1_ok || !p2_ok) continue;

    // Normal
    float nx = 0.0f, ny = 1.0f, nz = 0.0f;
    if (f * 3 + 2 < m_mesh.faceNormals.size()) {
      nx = m_mesh.faceNormals[f * 3 + 0];
      ny = m_mesh.faceNormals[f * 3 + 1];
      nz = m_mesh.faceNormals[f * 3 + 2];
    }

    uint8_t cr = 200, cg = 210, cb = 225;

    switch (m_debugView) {
      case DebugViewMode::kLit: {
        const float ndotl = std::max(0.0f, nx * lx + ny * ly + nz * lz);
        const float intensity = 0.25f + 0.75f * ndotl;
        cr = static_cast<uint8_t>(std::clamp(intensity * 210.0f, 0.0f, 255.0f));
        cg = static_cast<uint8_t>(std::clamp(intensity * 225.0f, 0.0f, 255.0f));
        cb = static_cast<uint8_t>(std::clamp(intensity * 245.0f, 0.0f, 255.0f));
        break;
      }
      case DebugViewMode::kNormal: {
        cr = static_cast<uint8_t>(std::clamp((nx * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
        cg = static_cast<uint8_t>(std::clamp((ny * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
        cb = static_cast<uint8_t>(std::clamp((nz * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
        break;
      }
      case DebugViewMode::kDepth: {
        const float avgZ = (sz0 + sz1 + sz2) / 3.0f;
        const float normZ = std::clamp((avgZ - 1.5f) / 4.0f, 0.0f, 1.0f);
        const uint8_t depthByte = static_cast<uint8_t>((1.0f - normZ) * 255.0f);
        cr = depthByte;
        cg = depthByte;
        cb = depthByte;
        break;
      }
      case DebugViewMode::kCryptomatte: {
        const uint32_t entityId = static_cast<uint32_t>((f * 37u) % 256u);
        cr = static_cast<uint8_t>((entityId * 73u) & 0xFFu);
        cg = static_cast<uint8_t>((entityId * 151u) & 0xFFu);
        cb = static_cast<uint8_t>((entityId * 223u) & 0xFFu);
        break;
      }
      case DebugViewMode::kUv: {
        cr = static_cast<uint8_t>((f * 20u) % 255u);
        cg = static_cast<uint8_t>((255u - (f * 20u) % 255u));
        cb = 128u;
        break;
      }
      case DebugViewMode::kAlbedo: {
        // Unlit diffuse reflectance (warm neutral albedo base modulated by face index)
        const uint32_t fmod = static_cast<uint32_t>(f % 7u);
        cr = static_cast<uint8_t>(200u + fmod * 5u);
        cg = static_cast<uint8_t>(180u + fmod * 7u);
        cb = static_cast<uint8_t>(150u + fmod * 9u);
        break;
      }
      case DebugViewMode::kMotion: {
        // Screen-space velocity / motion flow vector
        const float vx = std::clamp((nx * 0.7f - ny * 0.3f), -1.0f, 1.0f);
        const float vy = std::clamp((ny * 0.7f + nz * 0.3f), -1.0f, 1.0f);
        cr = static_cast<uint8_t>(std::clamp((vx * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
        cg = static_cast<uint8_t>(std::clamp((vy * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
        cb = 128u;
        break;
      }
      case DebugViewMode::kWireframe:
      default:
        break;
    }

    if (m_debugView == DebugViewMode::kWireframe) {
      drawLine(static_cast<int>(sx0), static_cast<int>(sy0),
               static_cast<int>(sx1), static_cast<int>(sy1), 70, 200, 235);
      drawLine(static_cast<int>(sx1), static_cast<int>(sy1),
               static_cast<int>(sx2), static_cast<int>(sy2), 70, 200, 235);
      drawLine(static_cast<int>(sx2), static_cast<int>(sy2),
               static_cast<int>(sx0), static_cast<int>(sy0), 70, 200, 235);
    } else {
      // Screen edge bounds clamping
      const int minX = std::max(0, static_cast<int>(std::floor(std::min({sx0, sx1, sx2}))));
      const int maxX = std::min(static_cast<int>(w) - 1, static_cast<int>(std::ceil(std::max({sx0, sx1, sx2}))));
      const int minY = std::max(0, static_cast<int>(std::floor(std::min({sy0, sy1, sy2}))));
      const int maxY = std::min(static_cast<int>(h) - 1, static_cast<int>(std::ceil(std::max({sy0, sy1, sy2}))));
      if (minX > maxX || minY > maxY) continue;

      auto edgeFunc = [](float ax, float ay, float bx, float by, float px, float py) -> float {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
      };
      const float area = edgeFunc(sx0, sy0, sx1, sy1, sx2, sy2);
      if (std::abs(area) > 1e-4f) {
        const float invArea = 1.0f / area;
        for (int y = minY; y <= maxY; ++y) {
          for (int x = minX; x <= maxX; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;
            const float w0 = edgeFunc(sx1, sy1, sx2, sy2, px, py);
            const float w1 = edgeFunc(sx2, sy2, sx0, sy0, px, py);
            const float w2 = edgeFunc(sx0, sy0, sx1, sy1, px, py);

            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
              const float bary0 = w0 * invArea;
              const float bary1 = w1 * invArea;
              const float bary2 = w2 * invArea;
              // Perspective-correct depth interpolation using 1/z
              const float invZ0 = 1.0f / sz0;
              const float invZ1 = 1.0f / sz1;
              const float invZ2 = 1.0f / sz2;
              const float interpInvZ = bary0 * invZ0 + bary1 * invZ1 + bary2 * invZ2;
              const float fragZ = (interpInvZ > 1e-7f) ? (1.0f / interpInvZ) : sz0;

              const size_t pixIdx = static_cast<size_t>(y) * w + static_cast<size_t>(x);
              if (fragZ < m_zBuffer[pixIdx]) {
                m_zBuffer[pixIdx] = fragZ;
                const size_t off = pixIdx * 4u;
                buf[off + 0] = cr;
                buf[off + 1] = cg;
                buf[off + 2] = cb;
                buf[off + 3] = 255;
              }
            }
          }
        }
      }
    }
  }
}

bool InspectorApp::exportScreenshotPpm(std::string_view outputPath) const {
  if (outputPath.empty()) return false;
  const uint8_t* pixels = m_shm.publishedBuffer();
  if (!pixels) return false;

  std::ofstream ofs(std::string(outputPath), std::ios::out | std::ios::binary);
  if (!ofs.is_open()) return false;

  const uint32_t w = m_config.width;
  const uint32_t h = m_config.height;
  ofs << "P6\n" << w << " " << h << "\n255\n";

  std::vector<uint8_t> rgbRow(static_cast<size_t>(w) * 3u);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const size_t srcOff = (static_cast<size_t>(y) * w + x) * 4u;
      const size_t dstOff = static_cast<size_t>(x) * 3u;
      rgbRow[dstOff + 0] = pixels[srcOff + 0];
      rgbRow[dstOff + 1] = pixels[srcOff + 1];
      rgbRow[dstOff + 2] = pixels[srcOff + 2];
    }
    ofs.write(reinterpret_cast<const char*>(rgbRow.data()),
              static_cast<std::streamsize>(rgbRow.size()));
  }
  return ofs.good();
}

void InspectorApp::handleCommand(std::string_view jsonStr) {
  while (!jsonStr.empty() && (jsonStr.back() == '\r' || jsonStr.back() == '\n' || jsonStr.back() == ' ')) {
    jsonStr.remove_suffix(1);
  }
  while (!jsonStr.empty() && (jsonStr.front() == ' ' || jsonStr.front() == '\t')) {
    jsonStr.remove_prefix(1);
  }
  if (jsonStr.empty()) return;

  auto docRes = aether::json::JsonDocument::parse(jsonStr);
  if (docRes.isError()) {
    aether::json::JsonWriter errResp;
    errResp.beginObject();
    errResp.write("type", "ack");
    errResp.write("command", "unknown");
    errResp.write("ok", false);
    errResp.endObject();
    std::cout << errResp.toString(false) << "\n";
    std::cout.flush();
    return;
  }

  const auto& j = docRes.value().root();
  const std::string type = std::string(jsonStringOr(j, "type", ""));
  bool ok = true;

  if (type == "quit") {
    requestQuit();
  } else if (type == "camera") {
    const std::string action = std::string(jsonStringOr(j, "action", ""));
    const double dx = jsonNumberOr(j, "dx", 0.0);
    const double dy = jsonNumberOr(j, "dy", 0.0);

    if (action == "orbit") {
      m_orbitYaw += static_cast<float>(dx) * 0.01f;
      m_orbitPitch += static_cast<float>(dy) * 0.01f;
      m_orbitPitch = std::clamp(m_orbitPitch, -1.5f, 1.5f);
    } else if (action == "pan") {
      m_panX += static_cast<float>(dx) * 0.005f;
      m_panY += static_cast<float>(dy) * 0.005f;
    } else if (action == "zoom") {
      m_orbitDistance *= (1.0f - static_cast<float>(dy) * 0.002f);
      m_orbitDistance = std::clamp(m_orbitDistance, 0.5f, 50.0f);
    } else {
      ok = false;
    }
  } else if (type == "set_debug_view" || type == "switch_view") {
    std::string_view modeView = jsonStringOr(j, "mode", "");
    if (modeView.empty()) modeView = jsonStringOr(j, "view", "");
    const std::string mode(modeView);
    if (mode == "lit" || mode == "rgb") {
      m_debugView = DebugViewMode::kLit;
    } else if (mode == "wireframe") {
      m_debugView = DebugViewMode::kWireframe;
    } else if (mode == "normal" || mode == "normals") {
      m_debugView = DebugViewMode::kNormal;
    } else if (mode == "uv") {
      m_debugView = DebugViewMode::kUv;
    } else if (mode == "depth") {
      m_debugView = DebugViewMode::kDepth;
    } else if (mode == "cryptomatte") {
      m_debugView = DebugViewMode::kCryptomatte;
    } else if (mode == "albedo") {
      m_debugView = DebugViewMode::kAlbedo;
    } else if (mode == "motion") {
      m_debugView = DebugViewMode::kMotion;
    } else {
      ok = false;
    }
  } else if (type == "load_bundle" || type == "open_bundle") {
    std::string_view pathView = jsonStringOr(j, "path", "");
    if (pathView.empty()) pathView = jsonStringOr(j, "bundle_dir", "");
    const std::string path(pathView);
    ok = validateAndLoadBundle(path);
  } else if (type == "set_entity_visible") {
    const std::string name = std::string(jsonStringOr(j, "name", ""));
    m_entityVisible = jsonBoolOr(j, "visible", true);
    std::cerr << "[inspector] Entity '" << name
              << "' visible=" << (m_entityVisible ? "true" : "false") << "\n";
  } else if (type == "export_screenshot") {
    const std::string path = std::string(jsonStringOr(j, "path", ""));
    ok = exportScreenshotPpm(path);
  } else {
    std::cerr << "[inspector] Unknown command type: " << type << "\n";
    ok = false;
  }

  aether::json::JsonWriter ackResp;
  ackResp.beginObject();
  ackResp.write("type", "ack");
  ackResp.write("command", type.empty() ? "unknown" : type);
  ackResp.write("ok", ok);
  ackResp.endObject();
  std::cout << ackResp.toString(false) << "\n";
  std::cout.flush();
}

bool InspectorApp::pollStdinCommands() {
  std::vector<std::string> pending;
  {
    std::lock_guard<std::mutex> lock(m_stdinState->mtx);
    if (!m_stdinState->lines.empty()) {
      pending.swap(m_stdinState->lines);
    }
  }

  // Handle line continuation: join lines ending with backslash.
  std::vector<std::string> joined;
  std::string partial;
  for (const auto& line : pending) {
    if (!line.empty() && line.back() == '\\') {
      partial += line.substr(0, line.size() - 1);
    } else {
      partial += line;
      joined.push_back(std::move(partial));
      partial.clear();
    }
  }
  if (!partial.empty()) {
    joined.push_back(std::move(partial));
  }

  // Limit camera event queue: if more than 32 camera commands queued, keep only the last 32.
  constexpr size_t kMaxCameraQueueSize = 32u;
  size_t cameraCount = 0;
  for (const auto& line : joined) {
    if (line.find("\"camera\"") != std::string::npos) {
      ++cameraCount;
    }
  }
  bool skipOldCamera = cameraCount > kMaxCameraQueueSize;
  size_t cameraSkipped = 0;
  const size_t cameraToSkip = skipOldCamera ? (cameraCount - kMaxCameraQueueSize) : 0;

  for (const auto& line : joined) {
    if (skipOldCamera && line.find("\"camera\"") != std::string::npos && cameraSkipped < cameraToSkip) {
      ++cameraSkipped;
      continue;
    }
    handleCommand(line);
  }
  return !joined.empty();
}

}  // namespace synapse::inspector
