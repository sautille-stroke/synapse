/// @file inspector_shm_server.cc
/// @brief Shared-memory frame server — platform-specific implementation.

#include "inspector_shm_server.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static size_t computeTotalSize(uint32_t w, uint32_t h) {
  const size_t frameBytes = static_cast<size_t>(w) * h * 4;  // RGBA8
  return sizeof(InspectorShmHeader) + frameBytes * 2;         // double buffer
}

// ---------------------------------------------------------------------------
// InspectorShmServer
// ---------------------------------------------------------------------------

InspectorShmServer::~InspectorShmServer() { close(); }

bool InspectorShmServer::open(std::string_view name, uint32_t width,
                              uint32_t height) {
  close();

  if (width == 0 || height == 0) return false;

  m_name = std::string(name);
  m_width = width;
  m_height = height;
  m_size = computeTotalSize(width, height);

#ifdef _WIN32
  // Windows: Named shared memory via CreateFileMapping.
  // Strip leading '/' for Windows named objects.
  std::string winName = m_name;
  if (!winName.empty() && winName[0] == '/') {
    winName = winName.substr(1);
  }

  m_hMapping = CreateFileMappingA(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(m_size >> 32),
      static_cast<DWORD>(m_size & 0xFFFFFFFF), winName.c_str());
  if (!m_hMapping) return false;

  m_base = MapViewOfFile(m_hMapping, FILE_MAP_ALL_ACCESS, 0, 0, m_size);
  if (!m_base) {
    CloseHandle(m_hMapping);
    m_hMapping = nullptr;
    return false;
  }
#else
  // POSIX: shm_open + mmap.
  m_fd = shm_open(m_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (m_fd < 0) return false;

  if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
    ::close(m_fd);
    shm_unlink(m_name.c_str());
    m_fd = -1;
    return false;
  }

  m_base =
      mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
  if (m_base == MAP_FAILED) {
    m_base = nullptr;
    ::close(m_fd);
    shm_unlink(m_name.c_str());
    m_fd = -1;
    return false;
  }
#endif

  // Initialise the header.
  std::memset(m_base, 0, sizeof(InspectorShmHeader));
  auto* hdr = static_cast<InspectorShmHeader*>(m_base);
  hdr->magic = kInspectorShmMagic;
  hdr->version = kInspectorShmVersion;
  hdr->width = width;
  hdr->height = height;

  return true;
}

void InspectorShmServer::close() {
  if (!m_base) return;

#ifdef _WIN32
  UnmapViewOfFile(m_base);
  if (m_hMapping) {
    CloseHandle(m_hMapping);
    m_hMapping = nullptr;
  }
#else
  munmap(m_base, m_size);
  if (m_fd >= 0) {
    ::close(m_fd);
    shm_unlink(m_name.c_str());
    m_fd = -1;
  }
#endif

  m_base = nullptr;
  m_size = 0;
}

uint8_t* InspectorShmServer::frontBuffer() noexcept {
  if (!m_base) return nullptr;
  auto* hdr = static_cast<InspectorShmHeader*>(m_base);
  const uint32_t idx = hdr->write_idx;
  const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 4;
  return static_cast<uint8_t*>(m_base) + sizeof(InspectorShmHeader) +
         frameBytes * idx;
}

void InspectorShmServer::publishFrame(float fps, float deltaTime) {
  if (!m_base) return;
  auto* hdr = static_cast<InspectorShmHeader*>(m_base);

  hdr->fps = fps;
  hdr->delta_time = deltaTime;

  // Flip write_idx and bump frame_ready atomically.
  // Use reinterpret_cast to access as std::atomic — the layout matches.
  auto& atomicIdx =
      *reinterpret_cast<std::atomic<uint32_t>*>(&hdr->write_idx);
  auto& atomicReady =
      *reinterpret_cast<std::atomic<uint64_t>*>(&hdr->frame_ready);

  atomicIdx.store(1u - atomicIdx.load(std::memory_order_relaxed),
                  std::memory_order_release);
  atomicReady.fetch_add(1u, std::memory_order_release);
}

size_t InspectorShmServer::totalSize() const noexcept { return m_size; }
