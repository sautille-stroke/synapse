/// @file inspector_shm_server.cc
/// @brief Shared-memory frame server backed by @aether//pal:shm with strict Seqlock.

#include "inspector_shm_server.h"

#include <atomic>
#include <cstring>
#include <new>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

bool InspectorShmServer::openFile(std::string_view filePath, uint32_t width,
                                  uint32_t height) {
  close();

  if (width == 0 || height == 0 || width > kMaxDimension ||
      height > kMaxDimension || filePath.empty()) {
    return false;
  }

  const size_t requiredSize = computeTotalSize(width, height);
  const std::string pathStr(filePath);

#if defined(_WIN32)
  HANDLE hFile = ::CreateFileA(
      pathStr.c_str(), GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) {
    return false;
  }
  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(requiredSize);
  if (!::SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN) || !::SetEndOfFile(hFile)) {
    ::CloseHandle(hFile);
    return false;
  }
  HANDLE hMapping = ::CreateFileMappingA(
      hFile, nullptr, PAGE_READWRITE, li.HighPart, li.LowPart, nullptr);
  ::CloseHandle(hFile);
  if (!hMapping) {
    return false;
  }
  auto shmResult = aether::pal::SharedMemory::fromHandle(
      aether::pal::PlatformHandle(hMapping, aether::pal::HandleType::SharedMemory),
      requiredSize, aether::pal::ShmAccess::ReadWrite);
#else
  int fd = ::open(pathStr.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd < 0) {
    return false;
  }
  if (::ftruncate(fd, static_cast<off_t>(requiredSize)) != 0) {
    ::close(fd);
    return false;
  }
  auto shmResult = aether::pal::SharedMemory::fromHandle(
      aether::pal::PlatformHandle(fd, aether::pal::HandleType::File),
      requiredSize, aether::pal::ShmAccess::ReadWrite);
#endif

  if (!shmResult.isOk()) {
    return false;
  }

  m_shm = std::move(shmResult.value());
  m_width = width;
  m_height = height;
  m_stagingIdx = 0;
  m_unlinkOnClose = false;  // File-backed: do not unlink (file persists)

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

void InspectorShmServer::beginFrame() noexcept {
  if (!m_shm.isValid()) return;
  auto* hdr = m_shm.as<InspectorShmHeader>(0);
  // Triple-buffer isolation: the staging buffer (m_stagingIdx) is always
  // distinct from the most recently published buffer (writeIdx), so the
  // writer can fill it without racing readers. We still set the odd-lock
  // to signal to readers that a write is in progress, but only if no
  // write is currently in progress (avoid double-increment).
  uint64_t seq = hdr->frameReady.load(std::memory_order_acquire);
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
  // release fence and release store on frameReady below.
  hdr->fps.store(fps, std::memory_order_relaxed);
  hdr->deltaTime.store(deltaTime, std::memory_order_relaxed);
  hdr->debugView.store(debugView, std::memory_order_relaxed);

  // Publish the completed buffer index with relaxed store; the subsequent
  // release fence ensures writeIdx is visible to readers before frameReady.
  hdr->writeIdx.store(m_stagingIdx, std::memory_order_relaxed);

  // ARM64 fix: A full release fence between writeIdx and frameReady ensures
  // that a reader observing the new frameReady will also see the new writeIdx.
  // Without this, two independent release stores to different atomics may be
  // reordered on weakly-ordered architectures (ARM64/RISC-V).
  std::atomic_thread_fence(std::memory_order_release);

  // Advance Seqlock sequence to next even number (stable)
  uint64_t seq = hdr->frameReady.load(std::memory_order_relaxed);
  uint64_t nextEven = (seq | 1u) + 1u;
  hdr->frameReady.store(nextEven, std::memory_order_release);

  m_stagingIdx = (m_stagingIdx + 1u) % kShmBufferCount;
  return nextEven / 2u;
}

uint64_t InspectorShmServer::sequenceNumber() const noexcept {
  if (!m_shm.isValid()) return 0;
  const auto* hdr = m_shm.as<InspectorShmHeader>(0);
  return hdr->frameReady.load(std::memory_order_acquire);
}

}  // namespace synapse::inspector
