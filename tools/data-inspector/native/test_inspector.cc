/// @file test_inspector.cc
/// @brief Unit tests for Data Inspector native application and SHM frame server.

#include "inspector_app.h"
#include "inspector_shm_server.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace synapse::inspector;

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

  // Test AOV view switching
  const char* views[] = {"lit", "wireframe", "normal", "depth", "cryptomatte", "uv"};
  for (const char* v : views) {
    std::string cmd = std::string(R"({"type":"set_debug_view","mode":")") + v + R"("})";
    app.handleCommand(cmd);

    // Render frame for each view mode
    app.renderFrame(0.016f);
    app.shmServer().publishFrame(60.0f, 0.016f);

    const uint8_t* pixels = app.shmServer().publishedBuffer();
    ASSERT_NE(pixels, nullptr) << "Pixels must be valid for view: " << v;

    // Assert rendered pixels contain non-background content
    bool hasNonZero = false;
    for (size_t i = 0; i < cfg.width * cfg.height * 4; ++i) {
      if (pixels[i] != 0 && pixels[i] != 16) {
        hasNonZero = true;
        break;
      }
    }
    EXPECT_TRUE(hasNonZero) << "Rendered AOV frame must contain valid non-zero content for view: " << v;
  }

  // Test screenshot export
  const std::string tmpPpm = "test_viewport_screenshot.ppm";
  bool exportOk = app.exportScreenshotPpm(tmpPpm);
  EXPECT_TRUE(exportOk) << "exportScreenshotPpm failed";
  EXPECT_TRUE(std::filesystem::exists(tmpPpm)) << "PPM file does not exist";
  EXPECT_GT(std::filesystem::file_size(tmpPpm), 1000u) << "PPM file too small";
  std::filesystem::remove(tmpPpm);

  // Run the loop for remaining frames
  app.run();
}

// ---------------------------------------------------------------------------
// Bundle Manifest Loading
// ---------------------------------------------------------------------------

TEST(InspectorApp, BundleManifestLoading) {
  // Create temporary bundle directory
  const std::string bundleDir = "test_temp_bundle";
  std::filesystem::create_directories(bundleDir);

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

  {
    std::ofstream ofs(bundleDir + "/bundle.json");
    ofs << manifestContent;
  }

  // Create OBJ model
  std::filesystem::create_directories(bundleDir + "/source");
  {
    std::ofstream ofs(bundleDir + "/source/test_cube.obj");
    ofs << "v -0.5 -0.5 0.0\n"
        << "v  0.5 -0.5 0.0\n"
        << "v  0.0  0.5 0.0\n"
        << "f 1 2 3\n";
  }

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

  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);
  const uint8_t* pixels = app.shmServer().publishedBuffer();
  EXPECT_NE(pixels, nullptr);

  // Clean up
  std::filesystem::remove_all(bundleDir);
}

// ---------------------------------------------------------------------------
// OBJ Loader: Out-of-bounds, negative, quad/ngon indices
// ---------------------------------------------------------------------------

TEST(InspectorApp, ObjLoaderOutOfBoundsIndices) {
  const std::string bundleDir = "test_oob_bundle";
  std::filesystem::create_directories(bundleDir + "/source");

  {
    std::ofstream ofs(bundleDir + "/bundle.json");
    ofs << R"({"sources":{"model":{"path":"source/bad.obj"}}})";
  }

  // OBJ with out-of-bounds positive index (9999) and negative relative index
  {
    std::ofstream ofs(bundleDir + "/source/bad.obj");
    ofs << "v 0.0 0.0 0.0\n"
        << "v 1.0 0.0 0.0\n"
        << "v 0.5 1.0 0.0\n"
        << "v 0.5 0.5 1.0\n"
        << "f 1 2 9999\n"        // out-of-bounds: should be dropped
        << "f -4 -3 -2\n"        // negative relative: valid (resolves to 0 1 2)
        << "f 1 2 3 4\n";        // quad: should be fan-triangulated
  }

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

  std::filesystem::remove_all(bundleDir);
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
