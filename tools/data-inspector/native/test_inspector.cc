/// @file test_inspector.cc
/// @brief Unit tests for Data Inspector native application and SHM frame server.

#include "inspector_app.h"
#include "inspector_shm_server.h"

#include <gtest/gtest.h>

#include <aether/fs/file_utils.h>
#include <aether/fs/filesystem.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace synapse::inspector;

namespace {

// Helper to construct a valid minimal binary GLB buffer containing 1 triangle (3 vertices, 3 indices).
std::vector<uint8_t> createMinimalGlb() {
  // Triangle vertices: (0, 0, 0), (1, 0, 0), (0, 1, 0) -> 3 * 3 * 4 = 36 bytes
  const float verts[9] = {
      0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
  };
  // Indices: 0, 1, 2 (uint16) -> 3 * 2 = 6 bytes + 2 bytes padding = 8 bytes
  const uint16_t indices[3] = {0, 1, 2};

  std::vector<uint8_t> binData(44, 0);
  std::memcpy(binData.data(), verts, sizeof(verts));
  std::memcpy(binData.data() + 36, indices, sizeof(indices));

  std::string jsonStr = R"({
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 44}]
})";

  // Pad JSON chunk to 4-byte boundary with spaces
  while (jsonStr.size() % 4 != 0) {
    jsonStr.push_back(' ');
  }

  const uint32_t jsonLen = static_cast<uint32_t>(jsonStr.size());
  const uint32_t binLen = static_cast<uint32_t>(binData.size());
  const uint32_t totalLen = 12u + (8u + jsonLen) + (8u + binLen);

  std::vector<uint8_t> glb(totalLen);
  uint8_t* ptr = glb.data();

  // Header: magic ("glTF"), version 2, totalLength
  const uint32_t magic = 0x46546C67u;
  const uint32_t version = 2u;
  std::memcpy(ptr, &magic, 4);
  std::memcpy(ptr + 4, &version, 4);
  std::memcpy(ptr + 8, &totalLen, 4);
  ptr += 12;

  // Chunk 0: JSON (0x4E4F534A)
  const uint32_t jsonType = 0x4E4F534Au;
  std::memcpy(ptr, &jsonLen, 4);
  std::memcpy(ptr + 4, &jsonType, 4);
  std::memcpy(ptr + 8, jsonStr.data(), jsonLen);
  ptr += 8 + jsonLen;

  // Chunk 1: BIN (0x004E4942)
  const uint32_t binType = 0x004E4942u;
  std::memcpy(ptr, &binLen, 4);
  std::memcpy(ptr + 4, &binType, 4);
  std::memcpy(ptr + 8, binData.data(), binLen);

  return glb;
}

}  // namespace

// ---------------------------------------------------------------------------
// SHM Server Tests
// ---------------------------------------------------------------------------

TEST(InspectorShmServer, BasicOpenClosePublish) {
  InspectorShmServer server;
  const std::string shmName = "/test-shm-server-basics";
  const uint32_t width = 320;
  const uint32_t height = 240;

  bool ok = server.open(shmName, width, height, true);
  ASSERT_TRUE(ok) << "Failed to open SHM server";
  EXPECT_TRUE(server.isOpen());
  EXPECT_GE(server.totalSize(),
            sizeof(InspectorShmHeader) + static_cast<size_t>(width) * height * 4 * 3);

  uint8_t* front = server.frontBuffer();
  ASSERT_NE(front, nullptr) << "Front buffer must not be null";

  // Fill front buffer with test pattern
  std::memset(front, 0xAB, static_cast<size_t>(width) * height * 4);

  // Publish frame
  uint64_t frameNum = server.publishFrame(60.0f, 0.0166f, 1);
  EXPECT_EQ(frameNum, 1u) << "First frame should have index 1";

  const uint8_t* pub = server.publishedBuffer();
  ASSERT_NE(pub, nullptr) << "Published buffer must not be null";
  EXPECT_EQ(pub[0], 0xAB) << "Published buffer byte 0 must match written pattern";
  EXPECT_EQ(pub[width * height * 4 - 1], 0xAB) << "Published buffer end byte must match pattern";

  server.close();
  EXPECT_FALSE(server.isOpen());
}

TEST(InspectorShmServer, SeqlockTransitions) {
  InspectorShmServer server;
  const std::string shmName = "/test-shm-seqlock";
  ASSERT_TRUE(server.open(shmName, 160, 120, true));

  // Initial state: even (0) -> stable
  EXPECT_EQ(server.sequenceNumber(), 0u);
  EXPECT_EQ(server.sequenceNumber() & 1u, 0u);

  // frontBuffer() no longer calls beginFrame() — seqlock stays at 0 (stable)
  uint8_t* buf1 = server.frontBuffer();
  ASSERT_NE(buf1, nullptr);
  EXPECT_EQ(server.sequenceNumber(), 0u);
  EXPECT_EQ(server.sequenceNumber() & 1u, 0u);  // still even (stable)

  // publishFrame() calls beginFrame() internally then transitions to next even.
  // Sequence: 0 -> 1 (odd/write) -> 2 (even/stable)
  uint64_t f1 = server.publishFrame(60.0f, 0.016f, 0);
  EXPECT_EQ(f1, 1u);
  EXPECT_EQ(server.sequenceNumber(), 2u);
  EXPECT_EQ(server.sequenceNumber() & 1u, 0u);  // even!

  // Frame 2: frontBuffer() stays stable
  uint8_t* buf2 = server.frontBuffer();
  ASSERT_NE(buf2, nullptr);
  EXPECT_EQ(server.sequenceNumber(), 2u);
  EXPECT_EQ(server.sequenceNumber() & 1u, 0u);  // still even

  // Frame 2 publish: 2 -> 3 (odd) -> 4 (even)
  uint64_t f2 = server.publishFrame(59.9f, 0.016f, 1);
  EXPECT_EQ(f2, 2u);
  EXPECT_EQ(server.sequenceNumber(), 4u);
  EXPECT_EQ(server.sequenceNumber() & 1u, 0u);  // even!

  server.close();
}

TEST(InspectorShmServer, DimensionLimits) {
  InspectorShmServer server;
  // Overly large dimensions (> kMaxDimension = 8192) must be rejected
  EXPECT_FALSE(server.open("/test-shm-oob-dim", 10000, 100, true));
  EXPECT_FALSE(server.open("/test-shm-oob-dim", 100, 10000, true));
  EXPECT_FALSE(server.open("/test-shm-oob-dim", 0, 100, true));
  EXPECT_FALSE(server.open("/test-shm-oob-dim", 100, 0, true));
}

TEST(InspectorShmServer, FileBackedShmOpenPublish) {
  InspectorShmServer server;
  auto tmpFile = std::filesystem::temp_directory_path() / "test-insp-file-shm.bin";
  std::error_code ec;
  std::filesystem::remove(tmpFile, ec);

  const uint32_t width = 64;
  const uint32_t height = 64;
  bool ok = server.openFile(tmpFile.string(), width, height);
  ASSERT_TRUE(ok) << "openFile failed on: " << tmpFile.string();
  EXPECT_TRUE(server.isOpen());

  uint8_t* front = server.frontBuffer();
  ASSERT_NE(front, nullptr);
  std::memset(front, 0x5A, static_cast<size_t>(width) * height * 4);

  uint64_t frameNum = server.publishFrame(30.0f, 0.033f, 0);
  EXPECT_EQ(frameNum, 1u);

  const uint8_t* pub = server.publishedBuffer();
  ASSERT_NE(pub, nullptr);
  EXPECT_EQ(pub[0], 0x5A);

  server.close();
  EXPECT_FALSE(server.isOpen());
  std::filesystem::remove(tmpFile, ec);
}

// ---------------------------------------------------------------------------
// Inspector App Lifecycle & AOV Tests
// ---------------------------------------------------------------------------

TEST(InspectorApp, LifecycleAndAovs) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-app-aov";
  cfg.width = 320;
  cfg.height = 240;
  cfg.maxFrames = 2;
  cfg.headless = true;
  cfg.keepShm = false;

  InspectorApp app(cfg);
  bool initOk = app.init();
  ASSERT_TRUE(initOk) << "InspectorApp::init failed";

  // Test commands
  app.handleCommand(R"({"type":"camera","action":"orbit","dx":15.0,"dy":10.0})");
  app.handleCommand(R"({"type":"camera","action":"zoom","dy":-5.0})");
  app.handleCommand(R"({"type":"camera","action":"pan","dx":2.0,"dy":1.0})");

  // Test all 8 AOV debug views including Albedo and Motion
  const char* views[] = {
      "lit", "wireframe", "normal", "depth", "cryptomatte", "uv", "albedo", "motion"
  };
  for (const char* v : views) {
    std::string cmd = std::string(R"({"type":"set_debug_view","mode":")") + v + R"("})";
    app.handleCommand(cmd);

    // Render frame for each view mode
    app.renderFrame(0.016f);
    app.shmServer().publishFrame(60.0f, 0.016f);

    const uint8_t* pixels = app.shmServer().publishedBuffer();
    ASSERT_NE(pixels, nullptr) << "Pixels must be valid for view: " << v;

    // Assert rendered pixels contain non-background content (background is 16, 20, 28)
    bool hasNonZero = false;
    for (size_t i = 0; i < cfg.width * cfg.height * 4; ++i) {
      if (pixels[i] != 0 && pixels[i] != 16 && pixels[i] != 20 && pixels[i] != 28) {
        hasNonZero = true;
        break;
      }
    }
    EXPECT_TRUE(hasNonZero) << "Rendered AOV frame must contain valid non-zero content for view: " << v;
  }

  // Test screenshot export
  const std::string tmpPpm = ::testing::TempDir() + "/test_viewport_screenshot.ppm";
  bool exportOk = app.exportScreenshotPpm(tmpPpm);
  EXPECT_TRUE(exportOk) << "exportScreenshotPpm failed";
  EXPECT_TRUE(aether::fs::exists(tmpPpm)) << "PPM file does not exist";
  EXPECT_GT(aether::fs::getFileSize(tmpPpm).valueOr(0), 1000u) << "PPM file too small";
  (void)aether::fs::remove(tmpPpm);

  // Run the loop for remaining frames
  app.run();
}

TEST(InspectorApp, SafeJsonHandling) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-safe-json";
  cfg.width = 160;
  cfg.height = 120;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  // Malformed JSON strings must not crash or trigger abort
  app.handleCommand("{bad json syntax");
  app.handleCommand("");
  app.handleCommand("   \r\n  ");

  // Invalid types under -fno-exceptions must return ok:false rather than aborting
  app.handleCommand(R"({"type": 12345})");
  app.handleCommand(R"({"type": "camera", "action": false, "dx": "bad", "dy": null})");
  app.handleCommand(R"({"type": "set_debug_view", "mode": 9999})");
  app.handleCommand(R"({"type": "set_entity_visible", "visible": "not_a_bool"})");
  app.handleCommand(R"({"type": "nonexistent_command"})");

  app.renderFrame(0.016f);
  EXPECT_NE(app.shmServer().frontBuffer(), nullptr);
}

// ---------------------------------------------------------------------------
// Bundle Manifest Loading
// ---------------------------------------------------------------------------

TEST(InspectorApp, BundleManifestLoading) {
  const std::string bundleDir = ::testing::TempDir() + "/test_native_bundle_obj";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir);

  const std::string manifestContent = R"({
  "schema_version": "synapse.bundle.v1.0",
  "bundle_id": "test_native_001",
  "sources": {
    "source_model": {
      "path": "source/test_cube.obj",
      "media_type": "model/obj"
    }
  }
})";

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json", manifestContent);

  // Create OBJ model
  (void)aether::fs::createDirectories(bundleDir + "/source");
  (void)aether::fs::writeAllText(bundleDir + "/source/test_cube.obj",
      "v -0.5 -0.5 0.0\n"
      "v  0.5 -0.5 0.0\n"
      "v  0.0  0.5 0.0\n"
      "f 1 2 3\n");

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-bundle-load";
  cfg.bundlePath = bundleDir;
  cfg.width = 160;
  cfg.height = 120;
  cfg.maxFrames = 1;
  cfg.headless = true;

  InspectorApp app(cfg);
  bool ok = app.init();
  ASSERT_TRUE(ok) << "Failed to load bundle in app.init";
  EXPECT_TRUE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 3u);
  EXPECT_EQ(app.faceCount(), 1u);

  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);
  const uint8_t* pixels = app.shmServer().publishedBuffer();
  EXPECT_NE(pixels, nullptr);

  // Clean up
  (void)aether::fs::removeAll(bundleDir);
}

TEST(InspectorApp, GlbLoaderValidAndStub) {
  const std::string bundleDir = ::testing::TempDir() + "/test_native_bundle_glb";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  // 1. Create valid GLB
  const auto glbBytes = createMinimalGlb();
  std::ofstream ofs(bundleDir + "/source/model.glb", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(glbBytes.data()), static_cast<std::streamsize>(glbBytes.size()));
  ofs.close();

  const std::string manifestContent = R"({
  "schema_version": "inspection.v1.0",
  "bundle_id": "test_glb_001",
  "sources": {
    "source_model": "source/model.glb"
  }
})";
  (void)aether::fs::writeAllText(bundleDir + "/bundle.json", manifestContent);

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-glb-load";
  cfg.bundlePath = bundleDir;
  cfg.width = 160;
  cfg.height = 120;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());
  EXPECT_TRUE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 3u);
  EXPECT_EQ(app.faceCount(), 1u);

  // 2. Test 12-byte stub GLB (minimal fixture): falls back safely to default cube without crash
  std::ofstream stubOfs(bundleDir + "/source/stub.glb", std::ios::binary);
  const uint32_t stubHeader[3] = {0x46546C67u, 2u, 12u};
  stubOfs.write(reinterpret_cast<const char*>(stubHeader), 12);
  stubOfs.close();

  const std::string stubManifest = R"({
  "schema_version": "inspection.v1.0",
  "bundle_id": "test_stub_001",
  "sources": {
    "source_model": "source/stub.glb"
  }
})";
  (void)aether::fs::writeAllText(bundleDir + "/bundle.json", stubManifest);

  // Reload bundle
  app.handleCommand(R"({"type":"load_bundle","path":")" + bundleDir + R"("})");
  EXPECT_FALSE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 8u);  // default cube

  (void)aether::fs::removeAll(bundleDir);
}

TEST(InspectorApp, PathTraversalRejected) {
  const std::string bundleDir = ::testing::TempDir() + "/test_traversal_bundle";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir);

  const std::string manifestContent = R"({
  "sources": {
    "bad_parent": "../../etc/shadow.obj",
    "bad_abs": "/etc/passwd.obj"
  }
})";
  (void)aether::fs::writeAllText(bundleDir + "/bundle.json", manifestContent);

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-traversal";
  cfg.bundlePath = bundleDir;
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());
  // Path traversal rejected -> safely falls back to default cube
  EXPECT_FALSE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 8u);

  (void)aether::fs::removeAll(bundleDir);
}

// ---------------------------------------------------------------------------
// OBJ Loader: Out-of-bounds, negative, quad/ngon indices
// ---------------------------------------------------------------------------

TEST(InspectorApp, ObjLoaderOutOfBoundsIndices) {
  const std::string bundleDir = ::testing::TempDir() + "/test_oob_bundle";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json",
      R"({"sources":{"model":{"path":"source/bad.obj"}}})");

  // OBJ with out-of-bounds positive index (9999) and negative relative index
  (void)aether::fs::writeAllText(bundleDir + "/source/bad.obj",
      "v 0.0 0.0 0.0\n"
      "v 1.0 0.0 0.0\n"
      "v 0.5 1.0 0.0\n"
      "v 0.5 0.5 1.0\n"
      "f 1 2 9999\n"        // out-of-bounds: should be dropped
      "f -4 -3 -2\n"        // negative relative: valid (resolves to 0 1 2)
      "f 1 2 3 4\n");        // quad: should be fan-triangulated

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-oob";
  cfg.bundlePath = bundleDir;
  cfg.width = 64;
  cfg.height = 64;
  cfg.maxFrames = 1;
  cfg.headless = true;

  InspectorApp app(cfg);
  bool ok = app.init();
  ASSERT_TRUE(ok) << "init must not crash on OBJ with OOB indices";

  // Render a frame — must not SIGSEGV
  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);
  const uint8_t* pixels = app.shmServer().publishedBuffer();
  EXPECT_NE(pixels, nullptr);

  (void)aether::fs::removeAll(bundleDir);
}

// ---------------------------------------------------------------------------
// Clean Fast Destruction (no deadlock)
// ---------------------------------------------------------------------------

TEST(InspectorApp, CleanFastDestruction) {
  auto t0 = std::chrono::steady_clock::now();
  {
    InspectorConfig cfg;
    cfg.shmName = "/test-insp-destruct";
    cfg.width = 64;
    cfg.height = 64;
    cfg.headless = true;

    InspectorApp app(cfg);
    bool ok = app.init();
    ASSERT_TRUE(ok);
    app.requestQuit();
  }
  auto t1 = std::chrono::steady_clock::now();
  auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  EXPECT_LT(elapsedMs, 500) << "Destructor took too long (> 500ms), possible deadlock";
}

// ---------------------------------------------------------------------------
// GLB Loader: Integer Overflow & Malicious Input Tests (PR-136)
// ---------------------------------------------------------------------------

namespace {

/// Helper: build a GLB with crafted JSON accessor fields to test overflow guards.
/// The BIN chunk is a small valid buffer; the JSON declares oversized counts
/// that would cause heap OOB reads if not checked.
std::vector<uint8_t> createMaliciousGlb(const std::string& accessorJson) {
  // Minimal BIN chunk: 48 bytes (enough for 4 floats of VEC3 + 6 uint16 indices)
  std::vector<uint8_t> binData(48, 0);
  // Write 1 valid triangle vertex at offset 0
  const float verts[3] = {0.0f, 0.0f, 0.0f};
  std::memcpy(binData.data(), verts, sizeof(verts));

  std::string jsonStr = accessorJson;
  // Pad JSON to 4-byte alignment
  while (jsonStr.size() % 4 != 0) jsonStr.push_back(' ');

  const uint32_t jsonLen = static_cast<uint32_t>(jsonStr.size());
  const uint32_t binLen = static_cast<uint32_t>(binData.size());
  const uint32_t totalLen = 12u + (8u + jsonLen) + (8u + binLen);

  std::vector<uint8_t> glb(totalLen);
  uint8_t* ptr = glb.data();

  const uint32_t magic = 0x46546C67u;
  const uint32_t version = 2u;
  std::memcpy(ptr, &magic, 4);
  std::memcpy(ptr + 4, &version, 4);
  std::memcpy(ptr + 8, &totalLen, 4);
  ptr += 12;

  const uint32_t jsonType = 0x4E4F534Au;
  std::memcpy(ptr, &jsonLen, 4);
  std::memcpy(ptr + 4, &jsonType, 4);
  std::memcpy(ptr + 8, jsonStr.data(), jsonLen);
  ptr += 8 + jsonLen;

  const uint32_t binType = 0x004E4942u;
  std::memcpy(ptr, &binLen, 4);
  std::memcpy(ptr + 4, &binType, 4);
  std::memcpy(ptr + 8, binData.data(), binLen);

  return glb;
}

}  // namespace

TEST(InspectorApp, GlbOverflowIdxCount0x40000001) {
  // rawIdxCount = 0x40000001: multiplied by 4 (sizeof uint32) in uint32_t
  // yields 0x00000004 (wraps around), which would pass a naive size check.
  // The fix must reject this via kMaxMeshIndices (48M) upper bound.
  const std::string json = R"({
    "asset": {"version": "2.0"},
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    "accessors": [
      {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 1, "type": "VEC3"},
      {"bufferView": 1, "byteOffset": 0, "componentType": 5125, "count": 1073741825, "type": "SCALAR"}
    ],
    "bufferViews": [
      {"buffer": 0, "byteOffset": 0, "byteLength": 12},
      {"buffer": 0, "byteOffset": 12, "byteLength": 4}
    ],
    "buffers": [{"byteLength": 48}]
  })";

  const std::string bundleDir = ::testing::TempDir() + "/test_glb_overflow_idx";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  auto glbBytes = createMaliciousGlb(json);
  std::ofstream ofs(bundleDir + "/source/evil.glb", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(glbBytes.data()),
            static_cast<std::streamsize>(glbBytes.size()));
  ofs.close();

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json",
      R"({"sources":{"model":"source/evil.glb"}})");

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-overflow-idx";
  cfg.bundlePath = bundleDir;
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init()) << "init must not crash on overflow GLB";
  // Overflow GLB should be rejected -> falls back to default cube
  EXPECT_FALSE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 8u);  // default cube

  // Rendering must not SIGSEGV
  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);
  EXPECT_NE(app.shmServer().publishedBuffer(), nullptr);

  (void)aether::fs::removeAll(bundleDir);
}

TEST(InspectorApp, GlbOverflowByteOffsetWraparound) {
  // 64-bit byteOffset wraparound: posViewOffset + posAccOffset overflows
  // when both are near SIZE_MAX/2. The fix must use overflow-safe addition.
  const std::string json = R"({
    "asset": {"version": "2.0"},
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    "accessors": [
      {"bufferView": 0, "byteOffset": 9223372036854775807, "componentType": 5126, "count": 3, "type": "VEC3"}
    ],
    "bufferViews": [
      {"buffer": 0, "byteOffset": 9223372036854775807, "byteLength": 36}
    ],
    "buffers": [{"byteLength": 48}]
  })";

  const std::string bundleDir = ::testing::TempDir() + "/test_glb_offset_wrap";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  auto glbBytes = createMaliciousGlb(json);
  std::ofstream ofs(bundleDir + "/source/evil.glb", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(glbBytes.data()),
            static_cast<std::streamsize>(glbBytes.size()));
  ofs.close();

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json",
      R"({"sources":{"model":"source/evil.glb"}})");

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-offset-wrap";
  cfg.bundlePath = bundleDir;
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init()) << "init must not crash on byteOffset wraparound GLB";
  EXPECT_FALSE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 8u);

  app.renderFrame(0.016f);
  EXPECT_NE(app.shmServer().frontBuffer(), nullptr);

  (void)aether::fs::removeAll(bundleDir);
}

TEST(InspectorApp, GlbNegativeByteStride) {
  // Negative byteStride: must be rejected (the fix checks rawByteStride < 0).
  const std::string json = R"({
    "asset": {"version": "2.0"},
    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    "accessors": [
      {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
    ],
    "bufferViews": [
      {"buffer": 0, "byteOffset": 0, "byteLength": 36, "byteStride": -12}
    ],
    "buffers": [{"byteLength": 48}]
  })";

  const std::string bundleDir = ::testing::TempDir() + "/test_glb_neg_stride";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  auto glbBytes = createMaliciousGlb(json);
  std::ofstream ofs(bundleDir + "/source/evil.glb", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(glbBytes.data()),
            static_cast<std::streamsize>(glbBytes.size()));
  ofs.close();

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json",
      R"({"sources":{"model":"source/evil.glb"}})");

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-neg-stride";
  cfg.bundlePath = bundleDir;
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init()) << "init must not crash on negative byteStride GLB";
  EXPECT_FALSE(app.hasCustomMesh());
  EXPECT_EQ(app.vertexCount(), 8u);

  app.renderFrame(0.016f);
  EXPECT_NE(app.shmServer().frontBuffer(), nullptr);

  (void)aether::fs::removeAll(bundleDir);
}


// ---------------------------------------------------------------------------
// PR-146: Multi-Entity, Tabs, Resize, Export ACK Tests
// ---------------------------------------------------------------------------

TEST(InspectorApp, MultiEntitySlots) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-multi-entity";
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  // Default: no custom entities.
  EXPECT_EQ(app.entityCount(), 0u);
  EXPECT_EQ(app.findEntity("source"), nullptr);
  EXPECT_FALSE(app.isEntityVisible("source"));

  // After loading a bundle with a source mesh, an entity should exist.
  const std::string bundleDir = ::testing::TempDir() + "/test_multi_entity";
  (void)aether::fs::removeAll(bundleDir);
  (void)aether::fs::createDirectories(bundleDir + "/source");

  auto glbData = createMinimalGlb();
  std::ofstream ofs(bundleDir + "/source/model.glb", std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(glbData.data()),
            static_cast<std::streamsize>(glbData.size()));
  ofs.close();

  (void)aether::fs::writeAllText(bundleDir + "/bundle.json",
      R"({"sources":{"source_model":"source/model.glb"}})");

  app.handleCommand(R"({"type":"load_bundle","path":")" + bundleDir + R"("})");
  EXPECT_GE(app.entityCount(), 1u);
  EXPECT_NE(app.findEntity("source"), nullptr);
  EXPECT_TRUE(app.isEntityVisible("source"));

  // Toggle visibility.
  EXPECT_TRUE(app.setEntityVisible("source", false));
  EXPECT_FALSE(app.isEntityVisible("source"));
  EXPECT_TRUE(app.setEntityVisible("source", true));
  EXPECT_TRUE(app.isEntityVisible("source"));

  // Non-existent entity.
  EXPECT_FALSE(app.setEntityVisible("nonexistent", false));

  (void)aether::fs::removeAll(bundleDir);
}

TEST(InspectorApp, ActiveTabSwitching) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-tabs";
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  EXPECT_EQ(app.activeTab(), "source");

  app.setActiveTab("animation");
  EXPECT_EQ(app.activeTab(), "animation");

  app.setActiveTab("video");
  EXPECT_EQ(app.activeTab(), "video");

  app.setActiveTab("comparison");
  EXPECT_EQ(app.activeTab(), "comparison");

  // Invalid tab: stays at current.
  app.setActiveTab("invalid_tab");
  EXPECT_EQ(app.activeTab(), "comparison");

  app.setActiveTab("source");
  EXPECT_EQ(app.activeTab(), "source");
}

TEST(InspectorApp, ResizeViewport) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-resize";
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  // Same size: should succeed without reopening.
  EXPECT_TRUE(app.resizeViewport(64, 64));

  // Valid resize.
  EXPECT_TRUE(app.resizeViewport(128, 96));

  // After resize, rendering should not crash.
  app.renderFrame(0.016f);
  EXPECT_NE(app.shmServer().frontBuffer(), nullptr);

  // Invalid resize: zero or too large.
  EXPECT_FALSE(app.resizeViewport(0, 100));
  EXPECT_FALSE(app.resizeViewport(100, 0));
  EXPECT_FALSE(app.resizeViewport(9000, 100));
}

TEST(InspectorApp, HandleCommandSelectTab) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-cmd-tab";
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  app.handleCommand(R"({"type":"select_tab","tab":"video"})");
  EXPECT_EQ(app.activeTab(), "video");

  app.handleCommand(R"({"type":"select_tab","tab":"comparison"})");
  EXPECT_EQ(app.activeTab(), "comparison");
}

TEST(InspectorApp, HandleCommandResize) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-cmd-resize";
  cfg.width = 64;
  cfg.height = 64;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());

  app.handleCommand(R"({"type":"resize","width":256,"height":128})");
  // After resize, renderFrame should work without crash.
  app.renderFrame(0.016f);
  EXPECT_NE(app.shmServer().frontBuffer(), nullptr);
}

TEST(InspectorApp, ExportScreenshotPpmAck) {
  InspectorConfig cfg;
  cfg.shmName = "/test-insp-screenshot-ack";
  cfg.width = 32;
  cfg.height = 32;
  cfg.headless = true;

  InspectorApp app(cfg);
  ASSERT_TRUE(app.init());
  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);

  const std::string outPath = ::testing::TempDir() + "/test_screenshot.ppm";
  EXPECT_TRUE(app.exportScreenshotPpm(outPath));

  // Verify the file exists and has PPM header.
  std::ifstream ifs(outPath, std::ios::binary);
  ASSERT_TRUE(ifs.is_open());
  std::string header;
  std::getline(ifs, header);
  EXPECT_EQ(header, "P6");
  ifs.close();
  std::filesystem::remove(outPath);
}
