#pragma once
/// @file inspector_shm_server.h
/// @brief Shared-memory frame server for the Data Inspector native process.
///
/// Implements the same SHM protocol as fulgur explorer (v2):
///   [Header: 64 bytes] [Buffer0: W*H*4] [Buffer1: W*H*4]
///
/// The Electron host reads frames from the "back" buffer while the native
/// process writes to the "front" buffer, coordinated via atomic write_idx
/// and frame_ready fields.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

/// @brief Magic value identifying a valid InspectorShmServer header.
static constexpr uint32_t kInspectorShmMagic = 0x494E5350u;  // "INSP"

/// @brief SHM protocol version.
static constexpr uint32_t kInspectorShmVersion = 1u;

/// @brief Default shared memory region name.
static constexpr std::string_view kDefaultShmName = "/inspector-frame";

/// @brief Default ring buffer names for JSON command/event channels.
static constexpr std::string_view kCmdRingName = "/inspector-debug-cmd";
static constexpr std::string_view kEvtRingName = "/inspector-debug-evt";

/// @brief Default ring buffer capacity (64 KB).
static constexpr uint32_t kRingCapacity = 64u * 1024u;

/// @brief 64-byte cache-line-aligned header at the start of the SHM region.
struct InspectorShmHeader {
  uint32_t magic;       ///< Must be kInspectorShmMagic.
  uint32_t version;     ///< Protocol version (kInspectorShmVersion).
  uint32_t width;       ///< Frame width in pixels.
  uint32_t height;      ///< Frame height in pixels.
  uint32_t write_idx;   ///< 0 or 1 — buffer currently being written.
  uint32_t m_pad0;      ///< Alignment padding.
  uint64_t frame_ready; ///< Bumped after each completed frame write.
  float fps;            ///< Measured FPS.
  float delta_time;     ///< Last frame delta in seconds.
  uint8_t debug_view;   ///< Active debug view mode (0 = lit).
  uint8_t m_reserved[15]; ///< Pad to 64 bytes.
};

static_assert(sizeof(InspectorShmHeader) == 64,
              "InspectorShmHeader must be exactly 64 bytes");

/// @brief Server-side SHM manager that creates the shared memory region
///        and provides methods to publish rendered frames.
class InspectorShmServer {
 public:
  InspectorShmServer() = default;
  ~InspectorShmServer();

  // Non-copyable, non-movable.
  InspectorShmServer(const InspectorShmServer&) = delete;
  InspectorShmServer& operator=(const InspectorShmServer&) = delete;

  /// @brief Open (create) the shared memory region for the given resolution.
  /// @param name  SHM region name (e.g., "/inspector-frame").
  /// @param width  Frame width in pixels.
  /// @param height Frame height in pixels.
  /// @return true on success.
  [[nodiscard]] bool open(std::string_view name, uint32_t width,
                          uint32_t height);

  /// @brief Close and unlink the shared memory region.
  void close();

  /// @brief Whether the SHM region is currently open.
  [[nodiscard]] bool isOpen() const noexcept { return m_base != nullptr; }

  /// @brief Get a pointer to the "front" pixel buffer for writing.
  ///
  /// The caller should fill this buffer with RGBA8 data and then call
  /// @ref publishFrame() to flip buffers.
  [[nodiscard]] uint8_t* frontBuffer() noexcept;

  /// @brief Signal to the reader that a new frame is available.
  ///
  /// Flips write_idx and increments frame_ready atomically.
  /// @param fps       Current FPS measurement.
  /// @param deltaTime Last frame delta in seconds.
  void publishFrame(float fps, float deltaTime);

  /// @brief Total SHM size in bytes for the current resolution.
  [[nodiscard]] size_t totalSize() const noexcept;

 private:
  void* m_base = nullptr;    ///< mmap/MapViewOfFile base pointer.
  size_t m_size = 0;         ///< Total mapped size.
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  std::string m_name;        ///< SHM region name for cleanup.

#ifdef _WIN32
  void* m_hMapping = nullptr; ///< Windows file mapping handle.
#else
  int m_fd = -1;              ///< POSIX shared memory fd.
#endif
};
