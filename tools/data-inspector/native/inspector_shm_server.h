#pragma once
/// @file inspector_shm_server.h
/// @brief Shared-memory frame server for the Data Inspector native process.
///
/// Uses @aether//pal:shm (`aether::pal::SharedMemory`) with triple-buffering:
///   [Header: 64 bytes] [Buffer0: W*H*4] [Buffer1: W*H*4] [Buffer2: W*H*4]
///
/// Implements a strict Seqlock protocol:
/// - Even `frameReady` (2k): frame is stable and consistent for readers.
/// - Odd `frameReady` (2k+1): writer is actively writing pixels / updating metadata.
/// - Writer increments `frameReady` to odd before writing, then updates `writeIdx`
///   and publishes `frameReady` to the next even number (2k+2) with release semantics.

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

/// @brief Maximum supported width / height dimension to prevent integer overflow.
inline constexpr uint32_t kMaxDimension = 8192u;

/// @brief Default shared memory region name prefix.
inline constexpr std::string_view kDefaultShmName = "/inspector-frame";

/// @brief 64-byte cache-line-aligned header at the start of the SHM region.
struct InspectorShmHeader {
  uint32_t magic;                     ///< Must be kInspectorShmMagic (offset 0).
  uint32_t version;                   ///< Protocol version (offset 4).
  uint32_t width;                     ///< Frame width in pixels (offset 8).
  uint32_t height;                    ///< Frame height in pixels (offset 12).
  std::atomic<uint32_t> writeIdx;     ///< Buffer index (0..2) of latest completed frame (offset 16).
  uint32_t bufferCount;               ///< Number of frame buffers (offset 20).
  std::atomic<uint64_t> frameReady;   ///< Seqlock sequence: even=stable, odd=writing (offset 24).
  std::atomic<float> fps;             ///< Measured FPS (offset 32).
  std::atomic<float> deltaTime;       ///< Last frame delta in seconds (offset 36).
  std::atomic<uint8_t> debugView;     ///< Active debug view mode (offset 40).
  uint8_t reserved[23];              ///< Pad to 64 bytes (offset 41..63).
};

static_assert(sizeof(InspectorShmHeader) == 64,
              "InspectorShmHeader must be exactly 64 bytes");
static_assert(offsetof(InspectorShmHeader, magic) == 0,
              "magic offset must be 0");
static_assert(offsetof(InspectorShmHeader, version) == 4,
              "version offset must be 4");
static_assert(offsetof(InspectorShmHeader, width) == 8,
              "width offset must be 8");
static_assert(offsetof(InspectorShmHeader, height) == 12,
              "height offset must be 12");
static_assert(offsetof(InspectorShmHeader, writeIdx) == 16,
              "writeIdx offset must be 16");
static_assert(offsetof(InspectorShmHeader, bufferCount) == 20,
              "bufferCount offset must be 20");
static_assert(offsetof(InspectorShmHeader, frameReady) == 24,
              "frameReady offset must be 24");
static_assert(offsetof(InspectorShmHeader, fps) == 32,
              "fps offset must be 32");
static_assert(offsetof(InspectorShmHeader, deltaTime) == 36,
              "deltaTime offset must be 36");
static_assert(offsetof(InspectorShmHeader, debugView) == 40,
              "debugView offset must be 40");
static_assert(offsetof(InspectorShmHeader, reserved) == 41,
              "reserved offset must be 41");

static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "std::atomic<uint32_t> must be lock-free for cross-process SHM");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "std::atomic<uint64_t> must be lock-free for cross-process SHM");
static_assert(std::atomic<float>::is_always_lock_free,
              "std::atomic<float> must be lock-free for cross-process SHM");

/// @brief Server-side SHM manager backed by `aether::pal::SharedMemory`.
class InspectorShmServer {
 public:
  InspectorShmServer() = default;
  ~InspectorShmServer();

  InspectorShmServer(const InspectorShmServer&) = delete;
  InspectorShmServer& operator=(const InspectorShmServer&) = delete;

  /// @brief Open (create) the shared memory region for the given resolution.
  /// @param name          SHM region name (e.g., "/inspector-frame-1234").
  /// @param width         Frame width in pixels (1..kMaxDimension).
  /// @param height        Frame height in pixels (1..kMaxDimension).
  /// @param unlinkOnClose Whether to unlink the SHM object from the OS namespace on close().
  /// @return true on success.
  [[nodiscard]] bool open(std::string_view name, uint32_t width,
                          uint32_t height, bool unlinkOnClose = true);

  /// @brief Close and optionally unlink the shared memory region.
  void close() noexcept;

  /// @brief Whether the SHM region is currently open.
  [[nodiscard]] bool isOpen() const noexcept { return m_shm.isValid(); }

  /// @brief Begin frame writing: transitions Seqlock counter to odd (writing in progress).
  void beginFrame() noexcept;

  /// @brief Get a pointer to the current staging pixel buffer for writing.
  /// Automatically calls beginFrame() to guarantee odd sequence transition before writes.
  [[nodiscard]] uint8_t* frontBuffer() noexcept;

  /// @brief Get a const pointer to the most recently published pixel buffer.
  [[nodiscard]] const uint8_t* publishedBuffer() const noexcept;

  /// @brief Signal to the reader that the current staging frame is complete.
  /// Transitions Seqlock counter to next even number with release semantics.
  /// @param fps       Current FPS measurement.
  /// @param deltaTime Last frame delta in seconds.
  /// @param debugView Active debug view mode.
  /// @return Total number of frames published so far.
  uint64_t publishFrame(float fps, float deltaTime, uint8_t debugView = 0) noexcept;

  /// @brief Get current raw Seqlock sequence number (even = stable, odd = writing).
  [[nodiscard]] uint64_t sequenceNumber() const noexcept;

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
