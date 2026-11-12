/// @file inspector_shm_server.cc
/// @brief Shared-memory frame server backed by @aether//pal:shm.

#include "inspector_shm_server.h"

#include <cstring>
#include <utility>

namespace synapse::inspector {

namespace {

[[nodiscard]] size_t computeTotalSize(uint32_t w, uint32_t h) noexcept {
  const size_t frameBytes = static_cast<size_t>(w) * h * 4u;  // RGBA8
  return sizeof(InspectorShmHeader) + frameBytes * kShmBufferCount;
}

}  // namespace

InspectorShmServer::~InspectorShmServer() { close(); }

bool InspectorShmServer::open(std::string_view name, uint32_t width,
                              uint32_t height, bool unlinkOnClose) {
  close();

  if (width == 0 || height == 0 || name.empty()) return false;

  const size_t requiredSize = computeTotalSize(width, height);
  auto shmResult = aether::pal::SharedMemory::create(
      name, requiredSize, aether::pal::ShmCreate::CreateOrOpen,
      aether::pal::ShmAccess::ReadWrite);
  if (!shmResult.isOk()) {
    return false;
  }

  m_shm = std::move(shmResult.value());
  m_width = width;
  m_height = height;
  m_stagingIdx = 0;
  m_unlinkOnClose = unlinkOnClose;

  void* base = m_shm.data();
  if (!base || m_shm.size() < requiredSize) {
    close();
    return false;
  }

  std::memset(base, 0, requiredSize);
  auto* hdr = static_cast<InspectorShmHeader*>(base);
  hdr->magic = kInspectorShmMagic;
  hdr->version = kInspectorShmVersion;
  hdr->width = width;
  hdr->height = height;
  hdr->write_idx.store(0u, std::memory_order_relaxed);
  hdr->buffer_count = kShmBufferCount;
  hdr->frame_ready.store(0u, std::memory_order_release);
  hdr->fps = 0.0f;
  hdr->delta_time = 0.0f;
  hdr->debug_view = 0u;

  return true;
}

void InspectorShmServer::close() noexcept {
  if (!m_shm.isValid()) return;

  const std::string shmName(m_shm.name());
  m_shm.close();
  if (m_unlinkOnClose && !shmName.empty()) {
    (void)aether::pal::SharedMemory::unlink(shmName);
  }
  m_width = 0;
  m_height = 0;
  m_stagingIdx = 0;
}

uint8_t* InspectorShmServer::frontBuffer() noexcept {
  if (!m_shm.isValid()) return nullptr;
  const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 4u;
  return m_shm.as<uint8_t>(sizeof(InspectorShmHeader) +
                           frameBytes * m_stagingIdx);
}

const uint8_t* InspectorShmServer::publishedBuffer() const noexcept {
  if (!m_shm.isValid()) return nullptr;
  const auto* hdr = m_shm.as<InspectorShmHeader>(0);
  const uint32_t pubIdx =
      hdr->write_idx.load(std::memory_order_acquire) % kShmBufferCount;
  const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 4u;
  return m_shm.as<uint8_t>(sizeof(InspectorShmHeader) + frameBytes * pubIdx);
}

uint64_t InspectorShmServer::publishFrame(float fps, float deltaTime,
                                          uint8_t debugView) noexcept {
  if (!m_shm.isValid()) return 0;
  auto* hdr = m_shm.as<InspectorShmHeader>(0);

  hdr->fps = fps;
  hdr->delta_time = deltaTime;
  hdr->debug_view = debugView;

  hdr->write_idx.store(m_stagingIdx, std::memory_order_release);
  const uint64_t readyCount =
      hdr->frame_ready.fetch_add(1u, std::memory_order_acq_rel) + 1u;

  m_stagingIdx = (m_stagingIdx + 1u) % kShmBufferCount;
  return readyCount;
}

}  // namespace synapse::inspector
