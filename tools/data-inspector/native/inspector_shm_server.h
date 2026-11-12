#pragma once
/// @file inspector_shm_server.h
/// @brief Shared-memory frame server for the Data Inspector native process.
///
/// Uses @aether//pal:shm (`aether::pal::SharedMemory`) with triple-buffering:
///   [Header: 64 bytes] [Buffer0: W*H*4] [Buffer1: W*H*4] [Buffer2: W*H*4]
///
/// The native process renders into the staging buffer (`m_stagingIdx`), then
/// atomically publishes that buffer index into `header.write_idx` and bumps
/// `header.frame_ready` before advancing `m_stagingIdx` to the next slot in
/// the 3-buffer ring. This guarantees the reader never reads a buffer being
/// actively written.

#include <aether/pal/shared_memory.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace synapse::inspector {

/// @brief Magic value identifying a valid InspectorShmServer header ("INSP").
inline constexpr uint32_t kInspectorShmMagic = 0x494E5350u;

/// @brief SHM protocol version.
inline constexpr uint32_t kInspectorShmVersion = 1u;

/// @brief Number of ring frame buffers in shared memory (triple buffering).
inline constexpr uint32_t kShmBufferCount = 3u;

/// @brief Default shared memory region name prefix.
inline constexpr std::string_view kDefaultShmName = "/inspector-frame";

/// @brief 64-byte cache-line-aligned header at the start of the SHM region.
struct InspectorShmHeader {
  uint32_t magic;                     ///< Must be kInspectorShmMagic.
  uint32_t version;                   ///< Protocol version (kInspectorShmVersion).
  uint32_t width;                     ///< Frame width in pixels.
  uint32_t height;                    ///< Frame height in pixels.
  std::atomic<uint32_t> write_idx;    ///< Buffer index (0..2) of latest completed frame.
  uint32_t buffer_count;              ///< Number of frame buffers (kShmBufferCount = 3).
  std::atomic<uint64_t> frame_ready;  ///< Monotonically incremented after each completed frame.
  float fps;                          ///< Measured FPS.
  float delta_time;                   ///< Last frame delta in seconds.
  uint8_t debug_view;                 ///< Active debug view mode (0 = lit).
  uint8_t m_reserved[23];             ///< Pad to 64 bytes.
};

static_assert(sizeof(InspectorShmHeader) == 64,
              "InspectorShmHeader must be exactly 64 bytes");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "std::atomic<uint32_t> must be lock-free for cross-process SHM");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "std::atomic<uint64_t> must be lock-free for cross-process SHM");

/// @brief Server-side SHM manager backed by `aether::pal::SharedMemory`.
class InspectorShmServer {
 public:
  InspectorShmServer() = default;
  ~InspectorShmServer();

  InspectorShmServer(const InspectorShmServer&) = delete;
  InspectorShmServer& operator=(const InspectorShmServer&) = delete;

  /// @brief Open (create) the shared memory region for the given resolution.
  /// @param name          SHM region name (e.g., "/inspector-frame-1234").
  /// @param width         Frame width in pixels.
  /// @param height        Frame height in pixels.
  /// @param unlinkOnClose Whether to unlink the SHM object from the OS namespace on close().
  /// @return true on success.
  [[nodiscard]] bool open(std::string_view name, uint32_t width,
                          uint32_t height, bool unlinkOnClose = true);

  /// @brief Close and optionally unlink the shared memory region.
  void close() noexcept;

  /// @brief Whether the SHM region is currently open.
  [[nodiscard]] bool isOpen() const noexcept { return m_shm.isValid(); }

  /// @brief Get a pointer to the current staging pixel buffer for writing.
  [[nodiscard]] uint8_t* frontBuffer() noexcept;

  /// @brief Get a const pointer to the most recently published pixel buffer.
  [[nodiscard]] const uint8_t* publishedBuffer() const noexcept;

  /// @brief Signal to the reader that the current staging frame is complete.
  /// @param fps       Current FPS measurement.
  /// @param deltaTime Last frame delta in seconds.
  /// @param debugView Active debug view mode.
  /// @return Total number of frames published so far.
  uint64_t publishFrame(float fps, float deltaTime, uint8_t debugView = 0) noexcept;

  /// @brief Total SHM size in bytes for the current resolution.
  [[nodiscard]] size_t totalSize() const noexcept { return m_shm.size(); }

 private:
  aether::pal::SharedMemory m_shm;
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  uint32_t m_stagingIdx = 0;
  bool m_unlinkOnClose = true;
};

}  // namespace synapse::inspector
