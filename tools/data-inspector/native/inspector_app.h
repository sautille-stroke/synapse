#pragma once
/// @file inspector_app.h
/// @brief Data Inspector native application — headless rendering + SHM.
///
/// Loads an InspectionBundle, renders the source model using fulgur's
/// Null RHI backend (or a real GPU backend when available), and streams
/// frames to the Electron host via shared memory.

#include "inspector_shm_server.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

/// @brief Debug view modes supported by the inspector.
enum class DebugViewMode : uint8_t {
  kLit = 0,       ///< Standard PBR lit view.
  kWireframe = 1, ///< Wireframe overlay.
  kNormal = 2,    ///< World-space normal visualisation.
  kUv = 3,        ///< UV coordinate visualisation.
};

/// @brief Camera command types sent from the Electron host.
enum class CameraAction : uint8_t {
  kOrbit = 0,
  kPan = 1,
  kZoom = 2,
};

/// @brief Configuration for the inspector native process.
struct InspectorConfig {
  std::string bundlePath;              ///< Path to the InspectionBundle dir.
  std::string rhiBackend = "null";     ///< "vulkan", "metal", "d3d12", "null".
  std::string shmName = std::string(kDefaultShmName);
  uint32_t width = 1280;
  uint32_t height = 720;
  bool headless = true;                ///< Run without a visible window.
};

/// @brief Main application class for the Data Inspector native process.
///
/// Lifecycle:
///   1. Construct with InspectorConfig.
///   2. Call init() to load the bundle and set up the renderer.
///   3. Call run() to enter the frame loop (blocks until quit).
///   4. Destruction cleans up SHM and renderer resources.
class InspectorApp {
 public:
  explicit InspectorApp(InspectorConfig config);
  ~InspectorApp();

  // Non-copyable, non-movable.
  InspectorApp(const InspectorApp&) = delete;
  InspectorApp& operator=(const InspectorApp&) = delete;

  /// @brief Initialise the renderer and load the bundle's source model.
  /// @return true on success, false if the bundle or model could not be loaded.
  [[nodiscard]] bool init();

  /// @brief Run the main frame loop. Blocks until requestQuit() is called.
  void run();

  /// @brief Signal the frame loop to exit after the current frame.
  void requestQuit() noexcept { m_running = false; }

  /// @brief Process a JSON command from the Electron host.
  ///
  /// Supported commands:
  ///   {"type":"camera","action":"orbit","dx":1.0,"dy":0.0}
  ///   {"type":"set_debug_view","mode":"wireframe"}
  ///   {"type":"set_entity_visible","name":"...","visible":true}
  ///   {"type":"export_screenshot","path":"..."}
  ///   {"type":"load_bundle","path":"..."}
  ///   {"type":"quit"}
  void handleCommand(std::string_view jsonCmd);

 private:
  InspectorConfig m_config;
  InspectorShmServer m_shm;
  bool m_running = false;
  DebugViewMode m_debugView = DebugViewMode::kLit;

  // Camera orbit state.
  float m_orbitYaw = 0.0f;
  float m_orbitPitch = 0.3f;
  float m_orbitDistance = 3.0f;
  float m_panX = 0.0f;
  float m_panY = 0.0f;

  /// @brief Render one frame into the SHM front buffer.
  void renderFrame(float deltaTime);

  /// @brief Read commands from stdin (line-delimited JSON).
  void pollStdinCommands();
};
