#pragma once
/// @file inspector_app.h
/// @brief Data Inspector native application — headless diagnostic rasterizer + SHM.
///
/// Loads an InspectionBundle manifest, renders diagnostic viewport frames
/// (Lit, Wireframe, WorldNormal, UV, Depth, Cryptomatte, Albedo, Motion) into a
/// triple-buffered shared memory region via @aether//pal:shm with strict Seqlock,
/// and processes JSON-RPC commands over stdin without exceptions.

#include "inspector_shm_server.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace synapse::inspector {

/// @brief Debug view modes supported by the inspector.
enum class DebugViewMode : uint8_t {
  kLit = 0,         ///< Diagnostic lit shading view (default).
  kWireframe = 1,   ///< Wireframe mesh / grid overlay.
  kNormal = 2,      ///< World-space normal visualisation.
  kUv = 3,          ///< UV coordinate visualisation.
  kDepth = 4,       ///< Camera linear depth visualisation.
  kCryptomatte = 5, ///< Entity ID segmentation mask visualisation.
  kAlbedo = 6,            ///< Base color / diffuse reflectance visualisation.
  kMotion = 7,            ///< Screen-space velocity / motion flow visualisation.
  kRoughnessMetallic = 8, ///< Roughness (R) / Metallic (G) PBR AOV visualisation.
};

/// @brief Configuration for the inspector native process.
struct InspectorConfig {
  std::string bundlePath;           ///< Path to the InspectionBundle dir.
  std::string rhiBackend = "null";  ///< "vulkan", "metal", "d3d12", "null".
  std::string shmName = std::string(kDefaultShmName);
  std::string shmFilePath;          ///< Path for file-backed SHM mapping.
  uint32_t width = 1280;
  uint32_t height = 720;
  uint32_t maxFrames = 0;           ///< 0 = continuous loop; >0 = exit after N frames.
  bool headless = true;             ///< Run without a visible window.
  bool keepShm = false;             ///< Keep SHM segment after exit (for CLI/test verification).
  bool useShmFile = false;          ///< Use file-backed SHM instead of POSIX shm_open.
};

struct StdinQueueState;

/// @brief Simple CPU mesh container for wireframe & AOV rasterisation.
struct InspectorMesh {
  std::vector<float> vertices;      ///< 3 floats per vertex (x, y, z)
  std::vector<uint32_t> faces;      ///< 3 vertex indices per triangle
  std::vector<float> faceNormals;   ///< 3 floats per triangle face normal
  std::vector<float> uvs;           ///< 2 floats per vertex (u, v) (optional)
  std::vector<float> albedoColors;  ///< 3 floats per vertex or face (RGB) (optional)
};

/// @brief Named mesh entity slot (e.g. source, animation, reconstructed).
struct MeshEntity {
  std::string name;                 ///< Entity identifier ("source", "animation", "reconstructed")
  std::string filePath;             ///< Relative or resolved path to model asset
  InspectorMesh mesh;               ///< Geometry container
  bool visible = true;              ///< Independent visibility toggle
  size_t submeshCount = 1;          ///< Submesh partition count
  size_t materialCount = 1;         ///< Material slot count
};

/// @brief Decoded 15-channel FGPP OpenEXR AOV container.
struct FgppAovData {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> radiance;          ///< 3 floats/px (DirectRadiance R, G, B)
  std::vector<float> depth;             ///< 1 float/px (CameraLinearDepth Z in meters)
  std::vector<float> normal;            ///< 3 floats/px (WorldNormal X, Y, Z)
  std::vector<float> motion;            ///< 2 floats/px (MotionVector X, Y)
  std::vector<float> baseColor;         ///< 3 floats/px (BaseColor R, G, B)
  std::vector<float> roughnessMetallic; ///< 2 floats/px (Roughness, Metallic)
  std::vector<uint32_t> cryptomatte;    ///< 1 u32/px (Cryptomatte ID)
  bool hasData = false;
};

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

  /// @brief Direct access to SHM server (for testing).
  [[nodiscard]] InspectorShmServer& shmServer() noexcept { return m_shm; }
  [[nodiscard]] const InspectorShmServer& shmServer() const noexcept { return m_shm; }

  /// @brief Direct render a single frame into SHM (for testing).
  void renderFrame(float deltaTime);

  /// @brief Export current published buffer as a PPM image file.
  [[nodiscard]] bool exportScreenshotPpm(std::string_view outputPath) const;

  /// @brief Diagnostic accessors for testing and inspection.
  [[nodiscard]] size_t vertexCount() const noexcept { return m_mesh.vertices.size() / 3; }
  [[nodiscard]] size_t faceCount() const noexcept { return m_mesh.faces.size() / 3; }
  [[nodiscard]] DebugViewMode debugViewMode() const noexcept { return m_debugView; }
  [[nodiscard]] bool hasCustomMesh() const noexcept { return m_hasCustomMesh; }

  /// @brief Multi-entity mesh management accessors.
  [[nodiscard]] size_t entityCount() const noexcept { return m_entities.size(); }
  [[nodiscard]] const std::vector<MeshEntity>& entities() const noexcept { return m_entities; }
  [[nodiscard]] const MeshEntity* findEntity(std::string_view name) const noexcept;
  [[nodiscard]] MeshEntity* findEntity(std::string_view name) noexcept;
  bool setEntityVisible(std::string_view name, bool visible) noexcept;
  [[nodiscard]] bool isEntityVisible(std::string_view name) const noexcept;

  /// @brief Active navigation tab & current timeline frame index.
  [[nodiscard]] const std::string& activeTab() const noexcept { return m_activeTab; }
  void setActiveTab(std::string_view tab) noexcept;
  [[nodiscard]] uint32_t currentFrame() const noexcept { return m_currentFrame; }

  /// @brief Dynamically resize viewport and recreate SHM framebuffer.
  [[nodiscard]] bool resizeViewport(uint32_t width, uint32_t height);

  /// @brief FGPP 15-channel AOV container access.
  [[nodiscard]] const FgppAovData& fgppAov() const noexcept { return m_fgppAov; }

 private:
  InspectorConfig m_config;
  InspectorShmServer m_shm;
  std::atomic<bool> m_running{false};
  DebugViewMode m_debugView = DebugViewMode::kLit;
  bool m_entityVisible = true;
  std::string m_activeTab = "source";
  uint32_t m_currentFrame = 0;

  // Multi-entity mesh slots (source, animation, reconstructed).
  std::vector<MeshEntity> m_entities;

  // Active primary mesh container (for backward compatibility with tests/view).
  InspectorMesh m_mesh;
  bool m_hasCustomMesh = false;

  // Persistent depth buffer to eliminate heap reallocation per frame at 60 Hz.
  std::vector<float> m_zBuffer;

  // Camera orbit/pan/zoom state.
  float m_orbitYaw = 0.4f;
  float m_orbitPitch = 0.35f;
  float m_orbitDistance = 3.5f;
  float m_panX = 0.0f;
  float m_panY = 0.0f;

  // 15-channel FGPP AOV container.
  FgppAovData m_fgppAov;

  // AOV overlay from loaded .exr file (backward compatibility).
  std::vector<float> m_exrAovData;
  uint32_t m_exrAovWidth = 0;
  uint32_t m_exrAovHeight = 0;
  bool m_hasExrAov = false;

  std::shared_ptr<StdinQueueState> m_stdinState;
  std::thread m_stdinThread;

  [[nodiscard]] bool validateAndLoadBundle(std::string_view bundlePath);
  void initDefaultMesh();
  bool loadObjMesh(std::string_view objPath, InspectorMesh& outMesh);
  bool loadGlbMesh(std::string_view glbPath, InspectorMesh& outMesh);
  bool loadObjMesh(std::string_view objPath) { return loadObjMesh(objPath, m_mesh); }
  bool loadGlbMesh(std::string_view glbPath) { return loadGlbMesh(glbPath, m_mesh); }
  bool loadExrAov(const std::string& path);
  bool pollStdinCommands();
};

}  // namespace synapse::inspector
