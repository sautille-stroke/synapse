/// @file inspector_app.cc
/// @brief Data Inspector native application implementation.

#include "inspector_app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#include <windows.h>
#define STDIN_FILENO 0
static bool stdinHasData() {
  HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
  if (h == INVALID_HANDLE_VALUE) return false;
  DWORD avail = 0;
  if (!PeekNamedPipe(h, nullptr, 0, nullptr, &avail, nullptr)) return false;
  return avail > 0;
}
#else
#include <poll.h>
#include <unistd.h>
static bool stdinHasData() {
  struct pollfd pfd = {STDIN_FILENO, POLLIN, 0};
  return poll(&pfd, 1, 0) > 0;
}
#endif

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency).
// Parses simple flat JSON objects for command dispatch.
// ---------------------------------------------------------------------------

static std::string jsonGetString(std::string_view json,
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

static double jsonGetNumber(std::string_view json, std::string_view key,
                            double fallback = 0.0) {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return fallback;
  ++pos;
  while (pos < json.size() && json[pos] == ' ') ++pos;
  try {
    return std::stod(std::string(json.substr(pos, 32)));
  } catch (...) {
    return fallback;
  }
}

static bool jsonGetBool(std::string_view json, std::string_view key,
                        bool fallback = false) {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto pos = json.find(needle);
  if (pos == std::string_view::npos) return fallback;
  pos = json.find(':', pos + needle.size());
  if (pos == std::string_view::npos) return fallback;
  ++pos;
  while (pos < json.size() && json[pos] == ' ') ++pos;
  if (pos < json.size() && json[pos] == 't') return true;
  if (pos < json.size() && json[pos] == 'f') return false;
  return fallback;
}

// ---------------------------------------------------------------------------
// InspectorApp
// ---------------------------------------------------------------------------

InspectorApp::InspectorApp(InspectorConfig config)
    : m_config(std::move(config)) {}

InspectorApp::~InspectorApp() { m_shm.close(); }

bool InspectorApp::init() {
  // Open SHM region.
  if (!m_shm.open(m_config.shmName, m_config.width, m_config.height)) {
    std::cerr << "[inspector] Failed to open SHM region: "
              << m_config.shmName << "\n";
    return false;
  }

  std::cout << "[inspector] SHM opened: " << m_config.shmName << " ("
            << m_config.width << "x" << m_config.height << ", "
            << m_shm.totalSize() << " bytes)\n";
  std::cout << "[inspector] RHI backend: " << m_config.rhiBackend << "\n";
  std::cout << "[inspector] Headless: "
            << (m_config.headless ? "true" : "false") << "\n";

  if (!m_config.bundlePath.empty()) {
    std::cout << "[inspector] Bundle: " << m_config.bundlePath << "\n";
  }

  // TODO(PR-27): Initialise fulgur RHI device with the selected backend.
  // For the initial scaffold, we render a procedural test pattern into
  // the SHM buffer to validate the transport pipeline end-to-end.

  return true;
}

void InspectorApp::run() {
  m_running = true;

  using Clock = std::chrono::high_resolution_clock;
  auto lastTime = Clock::now();
  uint64_t frameCount = 0;
  float accumulatedTime = 0.0f;
  float measuredFps = 0.0f;

  std::cout << "[inspector] Entering frame loop\n";

  while (m_running) {
    const auto now = Clock::now();
    const float dt =
        std::chrono::duration<float>(now - lastTime).count();
    lastTime = now;

    accumulatedTime += dt;
    ++frameCount;
    if (accumulatedTime >= 1.0f) {
      measuredFps = static_cast<float>(frameCount) / accumulatedTime;
      frameCount = 0;
      accumulatedTime = 0.0f;
    }

    // Poll stdin for JSON commands from the Electron host.
    pollStdinCommands();

    // Render and publish.
    renderFrame(dt);
    m_shm.publishFrame(measuredFps, dt);

    // Target ~60 fps.
    const auto frameEnd = Clock::now();
    const float elapsed =
        std::chrono::duration<float>(frameEnd - now).count();
    const float sleepTime = (1.0f / 60.0f) - elapsed;
    if (sleepTime > 0.001f) {
      std::this_thread::sleep_for(
          std::chrono::duration<float>(sleepTime));
    }
  }

  std::cout << "[inspector] Frame loop exited\n";
}

void InspectorApp::renderFrame(float deltaTime) {
  uint8_t* buf = m_shm.frontBuffer();
  if (!buf) return;

  const uint32_t w = m_config.width;
  const uint32_t h = m_config.height;

  // Procedural test pattern: gradient + orbit-dependent highlight.
  // This validates the full SHM pipeline before real fulgur rendering
  // is integrated.
  const float yawNorm =
      (std::fmod(m_orbitYaw, 6.2832f) + 6.2832f) / 6.2832f;

  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const size_t offset = (static_cast<size_t>(y) * w + x) * 4;
      const float u = static_cast<float>(x) / static_cast<float>(w);
      const float v = static_cast<float>(y) / static_cast<float>(h);

      uint8_t r = 0, g = 0, b = 0;

      switch (m_debugView) {
        case DebugViewMode::kLit:
          // Dark blue gradient with orbit-dependent warm highlight.
          r = static_cast<uint8_t>(
              std::clamp(u * yawNorm * 180.0f, 0.0f, 255.0f));
          g = static_cast<uint8_t>(
              std::clamp(v * 80.0f, 0.0f, 255.0f));
          b = static_cast<uint8_t>(
              std::clamp((1.0f - v) * 200.0f + 30.0f, 0.0f, 255.0f));
          break;
        case DebugViewMode::kWireframe:
          // Grid pattern.
          if ((x % 32 < 2) || (y % 32 < 2)) {
            r = g = b = 200;
          } else {
            r = g = b = 20;
          }
          break;
        case DebugViewMode::kNormal:
          // RGB = XYZ normal (synthetic hemisphere).
          r = static_cast<uint8_t>(u * 255.0f);
          g = static_cast<uint8_t>(v * 255.0f);
          b = static_cast<uint8_t>(
              std::clamp(
                  std::sqrt(std::max(0.0f, 1.0f - u * u - v * v))
                      * 255.0f,
                  0.0f, 255.0f));
          break;
        case DebugViewMode::kUv:
          // UV checker pattern.
          r = static_cast<uint8_t>(u * 255.0f);
          g = static_cast<uint8_t>(v * 255.0f);
          b = 0;
          break;
      }

      buf[offset + 0] = r;
      buf[offset + 1] = g;
      buf[offset + 2] = b;
      buf[offset + 3] = 255;  // Alpha.
    }
  }
}

void InspectorApp::handleCommand(std::string_view json) {
  const std::string type = jsonGetString(json, "type");

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
    }
  } else if (type == "load_bundle") {
    const std::string path = jsonGetString(json, "path");
    if (!path.empty()) {
      m_config.bundlePath = path;
      std::cout << "[inspector] Loading bundle: " << path << "\n";
      // TODO(PR-27): Reload model from the new bundle path.
    }
  } else if (type == "set_entity_visible") {
    const std::string name = jsonGetString(json, "name");
    const bool visible = jsonGetBool(json, "visible", true);
    std::cout << "[inspector] Entity '" << name
              << "' visible=" << (visible ? "true" : "false") << "\n";
    // TODO(PR-27): Update entity visibility in the scene.
  } else if (type == "export_screenshot") {
    const std::string path = jsonGetString(json, "path");
    std::cout << "[inspector] Export screenshot to: " << path << "\n";
    // TODO(PR-27): Write current frame buffer to PNG at `path`.
  } else {
    std::cerr << "[inspector] Unknown command type: " << type << "\n";
  }

  // Emit JSON response for the command.
  std::cout << "{\"type\":\"ack\",\"command\":\"" << type << "\"}\n";
  std::cout.flush();
}

void InspectorApp::pollStdinCommands() {
  while (stdinHasData()) {
    std::string line;
    if (!std::getline(std::cin, line)) {
      // stdin closed — Electron host disconnected.
      requestQuit();
      return;
    }
    if (line.empty()) continue;
    // Strip trailing \r on Windows.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    handleCommand(line);
  }
}
