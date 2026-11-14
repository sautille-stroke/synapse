/// @file inspector_app.cc
/// @brief Data Inspector native application implementation.

#include "inspector_app.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace synapse::inspector {

struct StdinQueueState {
  std::mutex mtx;
  std::vector<std::string> lines;
  std::atomic<bool> eof{false};
  std::atomic<bool> stop{false};
};

namespace {

/// @brief Minimal single-level JSON string/number/bool extractors.
///
/// These handle flat JSON objects with unescaped values.  For production
/// use with nested payloads, consider nlohmann/json or simdjson.  This
/// implementation intentionally avoids external dependencies for the
/// lightweight diagnostic inspector binary.

[[nodiscard]] uint32_t fnv1aHash(std::string_view text) noexcept {
  uint32_t h = 2166136261u;
  for (unsigned char c : text) {
    h ^= static_cast<uint32_t>(c);
    h *= 16777619u;
  }
  return h;
}

[[nodiscard]] std::string jsonGetString(std::string_view json,
                                        std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) return {};
  pos = json.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return {};
  pos = json.find('"', pos + 1);
  if (pos == std::string_view::npos) return {};
  ++pos;
  auto end = json.find('"', pos);
  if (end == std::string_view::npos) return {};
  return std::string(json.substr(pos, end - pos));
}

[[nodiscard]] double jsonGetNumber(std::string_view json, std::string_view key,
                                   double fallback = 0.0) noexcept {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return fallback;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos >= json.size()) return fallback;

  const std::string token(json.substr(pos, 32));
  char* endPtr = nullptr;
  const double val = std::strtod(token.c_str(), &endPtr);
  if (endPtr == token.c_str() || !std::isfinite(val)) {
    return fallback;
  }
  return val;
}

[[nodiscard]] bool jsonGetBool(std::string_view json, std::string_view key,
                               bool fallback = false) noexcept {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return fallback;
  ++pos;
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
  if (pos < json.size() && json[pos] == 't') return true;
  if (pos < json.size() && json[pos] == 'f') return false;
  return fallback;
}

}  // namespace

InspectorApp::InspectorApp(InspectorConfig config)
    : m_config(std::move(config)),
      m_stdinState(std::make_shared<StdinQueueState>()) {}

InspectorApp::~InspectorApp() {
  requestQuit();
  if (m_stdinState) {
    m_stdinState->stop.store(true, std::memory_order_release);
  }
  // Close stdin's underlying buffer to unblock std::getline in the reader
  // thread.  This is safe because the inspector owns the stdin contract
  // (Electron host pipes commands through it) and we are shutting down.
  if (auto* buf = std::cin.rdbuf()) {
    std::cin.rdbuf(nullptr);
  }
  if (m_stdinThread.joinable()) {
    m_stdinThread.join();
  }
  m_shm.close();
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
    std::cerr << "[inspector] Bundle manifest not found: " << manifestPath
              << "\n";
    return false;
  }

  m_config.bundlePath = std::string(bundlePath);
  m_bundleSeed = fnv1aHash(bundlePath);
  std::cout << "[inspector] Loaded bundle: " << m_config.bundlePath
            << " (seed=" << m_bundleSeed << ")\n";
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
    std::cerr << "[inspector] Failed to open SHM region: " << m_config.shmName
              << "\n";
    return false;
  }

  std::cout << "[inspector] SHM opened: " << m_config.shmName << " ("
            << m_config.width << "x" << m_config.height << ", "
            << m_shm.totalSize() << " bytes)\n";
  std::cout << "[inspector] RHI backend: " << m_config.rhiBackend
            << " (diagnostic SHM rasterizer)\n";
  std::cout << "[inspector] Headless: "
            << (m_config.headless ? "true" : "false") << "\n";
  std::cout.flush();

  auto state = m_stdinState;
  m_stdinThread = std::thread([state]() {
    std::string line;
    while (!state->stop.load(std::memory_order_acquire) &&
           std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty()) continue;
      std::lock_guard<std::mutex> lock(state->mtx);
      state->lines.push_back(std::move(line));
    }
    state->eof.store(true, std::memory_order_release);
  });

  if (m_config.maxFrames > 0) {
    // Allow piped stdin commands in non-interactive tests to arrive before frame 1.
    for (int i = 0; i < 20; ++i) {
      if (m_stdinState->eof.load(std::memory_order_acquire)) break;
      {
        std::lock_guard<std::mutex> lock(m_stdinState->mtx);
        if (!m_stdinState->lines.empty()) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

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

  std::cout << "[inspector] Entering frame loop\n";
  std::cout.flush();

  while (m_running.load(std::memory_order_acquire)) {
    const auto now = Clock::now();
    const float dt = std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    accumulatedTime += dt;
    ++fpsFrameCounter;
    if (accumulatedTime >= 1.0f) {
      measuredFps = static_cast<float>(fpsFrameCounter) / accumulatedTime;
      fpsFrameCounter = 0;
      accumulatedTime = 0.0f;
    }

    const bool hadCommands = pollStdinCommands();
    if (!m_running.load(std::memory_order_acquire)) {
      break;
    }

    renderFrame(dt);
    const uint64_t readyCount = m_shm.publishFrame(
        measuredFps, dt, static_cast<uint8_t>(m_debugView));
    ++totalFrames;

    if (totalFrames == 1 || hadCommands || m_config.maxFrames > 0) {
      std::cout << "{\"type\":\"frame_ready\",\"frame\":" << readyCount
                << ",\"debug_view\":" << static_cast<int>(m_debugView) << "}\n";
      std::cout.flush();
    }

    if (m_config.maxFrames > 0 && totalFrames >= m_config.maxFrames) {
      requestQuit();
      break;
    }

    if (m_stdinState->eof.load(std::memory_order_acquire) &&
        m_config.maxFrames == 0) {
      // Stdin closed by host process — exit cleanly after draining commands.
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
    if (sleepTime > 0.001f && m_config.maxFrames == 0) {
      std::this_thread::sleep_for(std::chrono::duration<float>(sleepTime));
    }
  }

  std::cout << "[inspector] Frame loop exited\n";
  std::cout.flush();
}

void InspectorApp::renderFrame(float /*deltaTime*/) {
  uint8_t* buf = m_shm.frontBuffer();
  if (!buf) return;

  const uint32_t w = m_config.width;
  const uint32_t h = m_config.height;

  const float yawNorm =
      (std::fmod(m_orbitYaw, 6.2831853f) + 6.2831853f) / 6.2831853f;
  const float pitchNorm = std::clamp((m_orbitPitch + 1.5f) / 3.0f, 0.0f, 1.0f);
  const float zoomFactor = std::clamp(3.0f / m_orbitDistance, 0.2f, 4.0f);
  const uint8_t bundleTint = static_cast<uint8_t>((m_bundleSeed & 0x7Fu) + 24u);

  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const size_t offset = (static_cast<size_t>(y) * w + x) * 4u;
      const float u = std::clamp(
          (static_cast<float>(x) / static_cast<float>(w) - 0.5f) / zoomFactor +
              0.5f + m_panX,
          0.0f, 1.0f);
      const float v = std::clamp(
          (static_cast<float>(y) / static_cast<float>(h) - 0.5f) / zoomFactor +
              0.5f + m_panY,
          0.0f, 1.0f);

      uint8_t r = 12, g = 16, b = 24;

      if (m_entityVisible) {
        switch (m_debugView) {
          case DebugViewMode::kLit:
            r = static_cast<uint8_t>(std::clamp(
                u * (0.35f + 0.65f * yawNorm) * 200.0f + bundleTint * 0.25f,
                0.0f, 255.0f));
            g = static_cast<uint8_t>(std::clamp(
                v * (0.4f + 0.6f * pitchNorm) * 140.0f + 20.0f, 0.0f, 255.0f));
            b = static_cast<uint8_t>(std::clamp(
                (1.0f - v) * 200.0f + 35.0f, 0.0f, 255.0f));
            break;
          case DebugViewMode::kWireframe:
            if ((x % 16u < 2u) || (y % 16u < 2u)) {
              r = 210;
              g = 225;
              b = 245;
            } else {
              r = 18;
              g = 22;
              b = 30;
            }
            break;
          case DebugViewMode::kNormal: {
            const float nx = u * 2.0f - 1.0f;
            const float ny = v * 2.0f - 1.0f;
            const float nz =
                std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
            r = static_cast<uint8_t>(std::clamp((nx * 0.5f + 0.5f) * 255.0f,
                                                0.0f, 255.0f));
            g = static_cast<uint8_t>(std::clamp((ny * 0.5f + 0.5f) * 255.0f,
                                                0.0f, 255.0f));
            b = static_cast<uint8_t>(std::clamp((nz * 0.5f + 0.5f) * 255.0f,
                                                0.0f, 255.0f));
            break;
          }
          case DebugViewMode::kUv:
            r = static_cast<uint8_t>(u * 255.0f);
            g = static_cast<uint8_t>(v * 255.0f);
            b = ((static_cast<uint32_t>(u * 8.0f) ^
                  static_cast<uint32_t>(v * 8.0f)) &
                 1u)
                    ? 180u
                    : 30u;
            break;
        }
      }

      buf[offset + 0] = r;
      buf[offset + 1] = g;
      buf[offset + 2] = b;
      buf[offset + 3] = 255u;
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

void InspectorApp::handleCommand(std::string_view json) {
  const std::string type = jsonGetString(json, "type");
  bool ok = true;

  if (type == "quit") {
    requestQuit();
  } else if (type == "camera") {
    const std::string action = jsonGetString(json, "action");
    const double dx = jsonGetNumber(json, "dx");
    const double dy = jsonGetNumber(json, "dy");

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
  } else if (type == "set_debug_view") {
    const std::string mode = jsonGetString(json, "mode");
    if (mode == "lit") {
      m_debugView = DebugViewMode::kLit;
    } else if (mode == "wireframe") {
      m_debugView = DebugViewMode::kWireframe;
    } else if (mode == "normal") {
      m_debugView = DebugViewMode::kNormal;
    } else if (mode == "uv") {
      m_debugView = DebugViewMode::kUv;
    } else {
      ok = false;
    }
  } else if (type == "load_bundle") {
    const std::string path = jsonGetString(json, "path");
    ok = validateAndLoadBundle(path);
  } else if (type == "set_entity_visible") {
    const std::string name = jsonGetString(json, "name");
    m_entityVisible = jsonGetBool(json, "visible", true);
    std::cout << "[inspector] Entity '" << name
              << "' visible=" << (m_entityVisible ? "true" : "false") << "\n";
  } else if (type == "export_screenshot") {
    const std::string path = jsonGetString(json, "path");
    ok = exportScreenshotPpm(path);
  } else {
    std::cerr << "[inspector] Unknown command type: " << type << "\n";
    ok = false;
  }

  std::cout << "{\"type\":\"ack\",\"command\":\"" << type
            << "\",\"ok\":" << (ok ? "true" : "false") << "}\n";
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
