/// @file inspector_shm_server.cc
/// @brief Shared-memory frame server backed by @aether//pal:shm with strict Seqlock.

#include "inspector_shm_server.h"

#include <cstring>
#include <new>
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

  if (width == 0 || height == 0 || width > kMaxDimension ||
      height > kMaxDimension || name.empty()) {
    return false;
  }

  const size_t requiredSize = computeTotalSize(width, height);
  auto shmResult = aether::pal::SharedMemory::create(
      name, requiredSize, aether::pal::ShmCreate::CreateNew,
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
  auto* hdr = new (base) InspectorShmHeader();
  hdr->magic = kInspectorShmMagic;
  hdr->version = kInspectorShmVersion;
  hdr->width = width;
  hdr->height = height;
  hdr->writeIdx.store(0u, std::memory_order_relaxed);
  hdr->bufferCount = kShmBufferCount;
  hdr->frameReady.store(0u, std::memory_order_release);
  hdr->fps.store(0.0f, std::memory_order_relaxed);
  hdr->deltaTime.store(0.0f, std::memory_order_relaxed);
  hdr->debugView.store(0u, std::memory_order_relaxed);

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

void InspectorShmServer::beginFrame() noexcept {
  if (!m_shm.isValid()) return;
  auto* hdr = m_shm.as<InspectorShmHeader>(0);
  uint64_t seq = hdr->frameReady.load(std::memory_order_relaxed);
  if ((seq & 1u) == 0u) {
    // Current seq is even (stable) -> transition to odd (write in progress)
    hdr->frameReady.store(seq + 1u, std::memory_order_release);
  }
}

uint8_t* InspectorShmServer::frontBuffer() noexcept {
  if (!m_shm.isValid()) return nullptr;
  beginFrame();
  const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 4u;
  return m_shm.as<uint8_t>(sizeof(InspectorShmHeader) +
                           frameBytes * m_stagingIdx);
}

const uint8_t* InspectorShmServer::publishedBuffer() const noexcept {
  if (!m_shm.isValid()) return nullptr;
  const auto* hdr = m_shm.as<InspectorShmHeader>(0);
  const uint32_t pubIdx =
      hdr->writeIdx.load(std::memory_order_acquire) % kShmBufferCount;
  const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 4u;
  return m_shm.as<uint8_t>(sizeof(InspectorShmHeader) + frameBytes * pubIdx);
}

uint64_t InspectorShmServer::publishFrame(float fps, float deltaTime,
                                          uint8_t debugView) noexcept {
  if (!m_shm.isValid()) return 0;
  auto* hdr = m_shm.as<InspectorShmHeader>(0);

  // Store metadata with relaxed ordering — visibility is guaranteed by the
  // release store on frameReady below.
  hdr->fps.store(fps, std::memory_order_relaxed);
  hdr->deltaTime.store(deltaTime, std::memory_order_relaxed);
  hdr->debugView.store(debugView, std::memory_order_relaxed);

  // Publish the completed buffer index
  hdr->writeIdx.store(m_stagingIdx, std::memory_order_release);

  // Advance Seqlock sequence to next even number (stable)
  uint64_t seq = hdr->frameReady.load(std::memory_order_relaxed);
  if ((seq & 1u) != 0u) {
    seq += 1u;
  } else {
    seq += 2u;
  }
  hdr->frameReady.store(seq, std::memory_order_release);

  m_stagingIdx = (m_stagingIdx + 1u) % kShmBufferCount;
  return seq / 2u;
}

uint64_t InspectorShmServer::sequenceNumber() const noexcept {
  if (!m_shm.isValid()) return 0;
  const auto* hdr = m_shm.as<InspectorShmHeader>(0);
  return hdr->frameReady.load(std::memory_order_acquire);
}

}  // namespace synapse::inspector
