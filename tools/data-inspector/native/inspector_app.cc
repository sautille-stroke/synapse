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
  const std::string pathStr(glbPath);
  std::ifstream ifs(pathStr, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) return false;

  const std::streamsize fileSize = ifs.tellg();
  if (fileSize < 12) return false;

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
      const auto attrs = prim["attributes"];
      if (!attrs.isObject() || !attrs.hasKey("POSITION")) continue;

      const uint32_t posAccIdx = static_cast<uint32_t>(attrs["POSITION"].getIntOr(-1));
      if (posAccIdx >= accessors.size()) continue;

      const auto posAcc = accessors[posAccIdx];
      if (!posAcc.isObject()) continue;

      const uint32_t posCount = static_cast<uint32_t>(posAcc["count"].getIntOr(0));
      const uint32_t posCompType = static_cast<uint32_t>(posAcc["componentType"].getIntOr(0));
      const uint32_t posViewIdx = static_cast<uint32_t>(posAcc["bufferView"].getIntOr(-1));
      const uint32_t posAccOffset = static_cast<uint32_t>(posAcc["byteOffset"].getIntOr(0));

      // POSITION must be FLOAT (5126)
      if (posCompType != 5126u || posCount == 0 || posViewIdx >= bufferViews.size()) continue;

      const auto posView = bufferViews[posViewIdx];
      if (!posView.isObject()) continue;

      const size_t posViewOffset = static_cast<size_t>(posView["byteOffset"].getIntOr(0));
      const size_t posByteOffset = posViewOffset + posAccOffset;
      const size_t posByteLength = static_cast<size_t>(posCount) * 3u * sizeof(float);

      if (binChunk.empty() || posByteOffset + posByteLength > binChunk.size()) continue;

      const uint32_t baseVertex = static_cast<uint32_t>(loadedVerts.size() / 3);
      const float* vertData = reinterpret_cast<const float*>(binChunk.data() + posByteOffset);

      bool vertsOk = true;
      for (size_t i = 0; i < posCount * 3; ++i) {
        if (!std::isfinite(vertData[i])) {
          vertsOk = false;
          break;
        }
      }
      if (!vertsOk) continue;

      loadedVerts.insert(loadedVerts.end(), vertData, vertData + posCount * 3);

      // Load indices if present
      if (prim.hasKey("indices")) {
        const uint32_t idxAccIdx = static_cast<uint32_t>(prim["indices"].getIntOr(-1));
        if (idxAccIdx < accessors.size()) {
          const auto idxAcc = accessors[idxAccIdx];
          const uint32_t idxCount = static_cast<uint32_t>(idxAcc["count"].getIntOr(0));
          const uint32_t idxCompType = static_cast<uint32_t>(idxAcc["componentType"].getIntOr(0));
          const uint32_t idxViewIdx = static_cast<uint32_t>(idxAcc["bufferView"].getIntOr(-1));
          const uint32_t idxAccOffset = static_cast<uint32_t>(idxAcc["byteOffset"].getIntOr(0));

          if (idxViewIdx < bufferViews.size()) {
            const auto idxView = bufferViews[idxViewIdx];
            const size_t idxViewOffset = static_cast<size_t>(idxView["byteOffset"].getIntOr(0));
            const size_t idxByteOffset = idxViewOffset + idxAccOffset;

            if (idxCompType == 5121u && idxByteOffset + idxCount <= binChunk.size()) {
              // UNSIGNED_BYTE
              const uint8_t* idxData = binChunk.data() + idxByteOffset;
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                if (idxData[i] < posCount && idxData[i + 1] < posCount && idxData[i + 2] < posCount) {
                  loadedFaces.push_back(baseVertex + idxData[i]);
                  loadedFaces.push_back(baseVertex + idxData[i + 1]);
                  loadedFaces.push_back(baseVertex + idxData[i + 2]);
                }
              }
            } else if (idxCompType == 5123u && idxByteOffset + idxCount * 2 <= binChunk.size()) {
              // UNSIGNED_SHORT
              const uint16_t* idxData = reinterpret_cast<const uint16_t*>(binChunk.data() + idxByteOffset);
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                if (idxData[i] < posCount && idxData[i + 1] < posCount && idxData[i + 2] < posCount) {
                  loadedFaces.push_back(baseVertex + idxData[i]);
                  loadedFaces.push_back(baseVertex + idxData[i + 1]);
                  loadedFaces.push_back(baseVertex + idxData[i + 2]);
                }
              }
            } else if (idxCompType == 5125u && idxByteOffset + idxCount * 4 <= binChunk.size()) {
              // UNSIGNED_INT
              const uint32_t* idxData = reinterpret_cast<const uint32_t*>(binChunk.data() + idxByteOffset);
              for (size_t i = 0; i + 2 < idxCount; i += 3) {
                if (idxData[i] < posCount && idxData[i + 1] < posCount && idxData[i + 2] < posCount) {
                  loadedFaces.push_back(baseVertex + idxData[i]);
                  loadedFaces.push_back(baseVertex + idxData[i + 1]);
                  loadedFaces.push_back(baseVertex + idxData[i + 2]);
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

  if (!m_shm.open(m_config.shmName, m_config.width, m_config.height,
                  !m_config.keepShm)) {
    std::cerr << "[inspector] Failed to open SHM region: " << m_config.shmName << "\n";
    return false;
  }

  std::cerr << "[inspector] SHM opened: " << m_config.shmName << " ("
            << m_config.width << "x" << m_config.height << ", "
            << m_shm.totalSize() << " bytes)\n";
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
    return std::isfinite(outSx) && std::isfinite(outSy) &&
           outSx >= 0.0f && outSx < static_cast<float>(w) &&
           outSy >= 0.0f && outSy < static_cast<float>(h);
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
              const float fragZ = bary0 * sz0 + bary1 * sz1 + bary2 * sz2;

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
  for (const auto& line : pending) {
    handleCommand(line);
  }
  return !pending.empty();
}

}  // namespace synapse::inspector
