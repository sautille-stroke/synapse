/// @file inspector_app.cc
/// @brief Implementation of Data Inspector native application.

#include "inspector_app.h"

#include <aether/pal/console.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace synapse::inspector {

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
  // The reader thread uses pollStdinBytes with a short timeout, so it will
  // notice the stop flag promptly and exit. No permanent deadlock on join().
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
        loadedVerts.push_back(x);
        loadedVerts.push_back(y);
        loadedVerts.push_back(z);
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

bool InspectorApp::validateAndLoadBundle(std::string_view bundlePath) {
  if (bundlePath.empty()) return false;
  std::string manifestPath(bundlePath);
  if (!manifestPath.empty() && manifestPath.back() != '/' &&
      manifestPath.back() != '\\') {
    manifestPath += '/';
  }
  manifestPath += "bundle.json";

  std::ifstream ifs(manifestPath, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "[inspector] Bundle manifest not found: " << manifestPath << "\n";
    return false;
  }

  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  auto j = nlohmann::json::parse(content, nullptr, false);
  if (j.is_discarded()) {
    std::cerr << "[inspector] Malformed bundle manifest: " << manifestPath << "\n";
    return false;
  }

  m_config.bundlePath = std::string(bundlePath);

  // Look for any .obj mesh referenced in sources
  if (j.contains("sources") && j["sources"].is_object()) {
    for (auto it = j["sources"].begin(); it != j["sources"].end(); ++it) {
      std::string relPath;
      if (it.value().is_string()) {
        relPath = it.value().get<std::string>();
      } else if (it.value().is_object() && it.value().contains("path")) {
        relPath = it.value()["path"].get<std::string>();
      }
      if (relPath.size() > 4 && relPath.substr(relPath.size() - 4) == ".obj") {
        std::string fullObjPath = std::string(bundlePath) + "/" + relPath;
        if (loadObjMesh(fullObjPath)) {
          std::cerr << "[inspector] Loaded mesh: " << fullObjPath << "\n";
          break;
        }
      }
    }
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

  auto state = m_stdinState;
  m_stdinThread = std::thread([state]() {
    // Line accumulation buffer for pollStdinBytes
    std::string lineBuf;
    constexpr size_t kChunkSize = 4096;
    char chunk[kChunkSize];

    while (!state->stop.load(std::memory_order_acquire)) {
      auto result = aether::pal::pollStdinBytes(chunk, kChunkSize);

      if (result.pipeClosed) {
        // Flush any trailing partial line
        if (!lineBuf.empty()) {
          if (!lineBuf.empty() && lineBuf.back() == '\r') {
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
        // No data available — sleep briefly to avoid busy-spin
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // Append received bytes and split on newlines
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
    nlohmann::json frameMsg;
    frameMsg["type"] = "frame_ready";
    std::cout << frameMsg.dump() << "\n";
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

  // Allocate z-buffer for real depth occlusion
  std::vector<float> zBuffer(pixelCount, std::numeric_limits<float>::max());

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
    if (zc <= 0.2f) return false;

    outSx = cx + (xc / zc) * fovScale;
    outSy = cy_screen - (yc / zc) * fovScale;
    return (outSx >= 0.0f && outSx < static_cast<float>(w) && outSy >= 0.0f && outSy < static_cast<float>(h));
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
      continue;  // Drop out-of-bounds face
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
        // Distinct entity colors based on face index
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
      // Simple raster scanline triangle fill with z-buffer depth occlusion
      int minX = std::max(0, static_cast<int>(std::min({sx0, sx1, sx2})));
      int maxX = std::min(static_cast<int>(w) - 1, static_cast<int>(std::max({sx0, sx1, sx2})));
      int minY = std::max(0, static_cast<int>(std::min({sy0, sy1, sy2})));
      int maxY = std::min(static_cast<int>(h) - 1, static_cast<int>(std::max({sy0, sy1, sy2})));

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
              // Interpolate depth for z-buffer test
              const float bary0 = w0 * invArea;
              const float bary1 = w1 * invArea;
              const float bary2 = w2 * invArea;
              const float fragZ = bary0 * sz0 + bary1 * sz1 + bary2 * sz2;

              const size_t pixIdx = static_cast<size_t>(y) * w + static_cast<size_t>(x);
              if (fragZ < zBuffer[pixIdx]) {
                zBuffer[pixIdx] = fragZ;
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
  // Strip trailing whitespace or carriage returns
  while (!jsonStr.empty() && (jsonStr.back() == '\r' || jsonStr.back() == '\n' || jsonStr.back() == ' ')) {
    jsonStr.remove_suffix(1);
  }
  while (!jsonStr.empty() && (jsonStr.front() == ' ' || jsonStr.front() == '\t')) {
    jsonStr.remove_prefix(1);
  }
  if (jsonStr.empty()) return;

  auto j = nlohmann::json::parse(jsonStr, nullptr, false);
  if (j.is_discarded()) {
    nlohmann::json errResp;
    errResp["type"] = "ack";
    errResp["command"] = "unknown";
    errResp["ok"] = false;
    std::cout << errResp.dump() << "\n";
    std::cout.flush();
    return;
  }

  const std::string type = j.value("type", "");
  bool ok = true;

  if (type == "quit") {
    requestQuit();
  } else if (type == "camera") {
    const std::string action = j.value("action", "");
    const double dx = j.value("dx", 0.0);
    const double dy = j.value("dy", 0.0);

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
    const std::string mode = j.value("mode", j.value("view", ""));
    if (mode == "lit" || mode == "rgb") {
      m_debugView = DebugViewMode::kLit;
    } else if (mode == "wireframe") {
      m_debugView = DebugViewMode::kWireframe;
    } else if (mode == "normal") {
      m_debugView = DebugViewMode::kNormal;
    } else if (mode == "uv") {
      m_debugView = DebugViewMode::kUv;
    } else if (mode == "depth") {
      m_debugView = DebugViewMode::kDepth;
    } else if (mode == "cryptomatte") {
      m_debugView = DebugViewMode::kCryptomatte;
    } else {
      ok = false;
    }
  } else if (type == "load_bundle" || type == "open_bundle") {
    const std::string path = j.value("path", j.value("bundle_dir", ""));
    ok = validateAndLoadBundle(path);
  } else if (type == "set_entity_visible") {
    const std::string name = j.value("name", "");
    m_entityVisible = j.value("visible", true);
    std::cerr << "[inspector] Entity '" << name
              << "' visible=" << (m_entityVisible ? "true" : "false") << "\n";
  } else if (type == "export_screenshot") {
    const std::string path = j.value("path", "");
    ok = exportScreenshotPpm(path);
  } else {
    std::cerr << "[inspector] Unknown command type: " << type << "\n";
    ok = false;
  }

  nlohmann::json ackResp;
  ackResp["type"] = "ack";
  ackResp["command"] = type;
  ackResp["ok"] = ok;
  std::cout << ackResp.dump() << "\n";
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
