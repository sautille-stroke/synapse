#pragma once
/// @file inspector_app.h
/// @brief Data Inspector native application — headless diagnostic rasterizer + SHM.
///
/// Loads an InspectionBundle manifest, renders diagnostic viewport frames
/// (Lit, Wireframe, WorldNormal, UV) into a triple-buffered shared memory
/// region via @aether//pal:shm, and processes JSON-RPC commands over stdin.

#include "inspector_shm_server.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace synapse::inspector {

/// @brief Debug view modes supported by the inspector.
enum class DebugViewMode : uint8_t {
  kLit = 0,        ///< Diagnostic lit shading view.
  kWireframe = 1,  ///< Wireframe grid overlay.
  kNormal = 2,     ///< World-space normal visualisation.
  kUv = 3,         ///< UV coordinate visualisation.
};

/// @brief Configuration for the inspector native process.
struct InspectorConfig {
  std::string bundlePath;           ///< Path to the InspectionBundle dir.
  std::string rhiBackend = "null";  ///< "vulkan", "metal", "d3d12", "null".
  std::string shmName = std::string(kDefaultShmName);
  uint32_t width = 1280;
  uint32_t height = 720;
  uint32_t maxFrames = 0;           ///< 0 = continuous loop; >0 = exit after N frames.
  bool headless = true;             ///< Run without a visible window.
  bool keepShm = false;             ///< Keep SHM segment after exit (for CLI/test verification).
};

struct StdinQueueState;

/// @brief Main application class for the Data Inspector native process.
class InspectorApp {
 public:
  explicit InspectorApp(InspectorConfig config);
  ~InspectorApp();

  InspectorApp(const InspectorApp&) = delete;
  InspectorApp& operator=(const InspectorApp&) = delete;

  /// @brief Initialise SHM and validate initial bundle path if provided.
  [[nodiscard]] bool init();

  /// @brief Run the main frame loop. Blocks until quit or maxFrames reached.
  void run();

  /// @brief Signal the frame loop to exit after the current frame.
  void requestQuit() noexcept { m_running.store(false, std::memory_order_release); }

  /// @brief Process a single JSON command line from the Electron host.
  void handleCommand(std::string_view jsonCmd);

 private:
  InspectorConfig m_config;
  InspectorShmServer m_shm;
  std::atomic<bool> m_running{false};
  DebugViewMode m_debugView = DebugViewMode::kLit;
  uint32_t m_bundleSeed = 0u;
  bool m_entityVisible = true;

  // Camera orbit/pan/zoom state.
  float m_orbitYaw = 0.0f;
  float m_orbitPitch = 0.3f;
  float m_orbitDistance = 3.0f;
  float m_panX = 0.0f;
  float m_panY = 0.0f;

  std::shared_ptr<StdinQueueState> m_stdinState;
  std::thread m_stdinThread;

  [[nodiscard]] bool validateAndLoadBundle(std::string_view bundlePath);
  [[nodiscard]] bool exportScreenshotPpm(std::string_view outputPath) const;
  void renderFrame(float deltaTime);
  bool pollStdinCommands();
};

}  // namespace synapse::inspector
