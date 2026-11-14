/// @file test_inspector.cc
/// @brief Unit tests for Data Inspector native application and SHM frame server.

#include "inspector_app.h"
#include "inspector_shm_server.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace synapse::inspector;

static void testShmServerBasics() {
  std::cout << "[test] Running testShmServerBasics...\n";

  InspectorShmServer server;
  const std::string shmName = "/test-shm-server-basics";
  const uint32_t width = 320;
  const uint32_t height = 240;

  bool ok = server.open(shmName, width, height, true);
  assert(ok && "Failed to open SHM server");
  assert(server.isOpen() && "SHM server should be open");
  assert(server.totalSize() >= sizeof(InspectorShmHeader) + static_cast<size_t>(width) * height * 4 * 3);

  uint8_t* front = server.frontBuffer();
  assert(front != nullptr && "Front buffer must not be null");

  // Fill front buffer with test pattern
  std::memset(front, 0xAB, static_cast<size_t>(width) * height * 4);

  // Publish frame
  uint64_t frameNum = server.publishFrame(60.0f, 0.0166f, 1);
  assert(frameNum == 1 && "First frame should have index 1");

  const uint8_t* pub = server.publishedBuffer();
  assert(pub != nullptr && "Published buffer must not be null");
  assert(pub[0] == 0xAB && "Published buffer byte 0 must match written pattern");
  assert(pub[width * height * 4 - 1] == 0xAB && "Published buffer end byte must match pattern");

  server.close();
  assert(!server.isOpen() && "SHM server should be closed");

  std::cout << "[test] testShmServerBasics PASSED\n";
}

static void testInspectorAppLifecycleAndAovs() {
  std::cout << "[test] Running testInspectorAppLifecycleAndAovs...\n";

  InspectorConfig cfg;
  cfg.shmName = "/test-insp-app-aov";
  cfg.width = 320;
  cfg.height = 240;
  cfg.maxFrames = 2;
  cfg.headless = true;
  cfg.keepShm = false;

  InspectorApp app(cfg);
  bool initOk = app.init();
  assert(initOk && "InspectorApp::init failed");

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
    assert(pixels != nullptr && "Pixels must be valid");

    // Assert rendered pixels contain non-background content
    bool hasNonZero = false;
    for (size_t i = 0; i < cfg.width * cfg.height * 4; ++i) {
      if (pixels[i] != 0 && pixels[i] != 16) {
        hasNonZero = true;
        break;
      }
    }
    assert(hasNonZero && "Rendered AOV frame must contain valid non-zero content");
  }

  // Test screenshot export
  const std::string tmpPpm = "test_viewport_screenshot.ppm";
  bool exportOk = app.exportScreenshotPpm(tmpPpm);
  assert(exportOk && "exportScreenshotPpm failed");
  assert(std::filesystem::exists(tmpPpm) && "PPM file does not exist");
  assert(std::filesystem::file_size(tmpPpm) > 1000 && "PPM file too small");
  std::filesystem::remove(tmpPpm);

  // Run the loop for remaining frames
  app.run();

  std::cout << "[test] testInspectorAppLifecycleAndAovs PASSED\n";
}

static void testBundleManifestLoading() {
  std::cout << "[test] Running testBundleManifestLoading...\n";

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
  assert(ok && "Failed to load bundle in app.init");

  app.renderFrame(0.016f);
  app.shmServer().publishFrame(60.0f, 0.016f);
  const uint8_t* pixels = app.shmServer().publishedBuffer();
  assert(pixels != nullptr);

  // Clean up
  std::filesystem::remove_all(bundleDir);

  std::cout << "[test] testBundleManifestLoading PASSED\n";
}

static void testCleanFastDestruction() {
  std::cout << "[test] Running testCleanFastDestruction...\n";

  auto t0 = std::chrono::steady_clock::now();
  {
    InspectorConfig cfg;
    cfg.shmName = "/test-insp-destruct";
    cfg.width = 64;
    cfg.height = 64;
    cfg.headless = true;

    InspectorApp app(cfg);
    app.init();
    app.requestQuit();
  }
  auto t1 = std::chrono::steady_clock::now();
  auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  std::cout << "[test] Clean destruction completed in " << elapsedMs << "ms\n";
  assert(elapsedMs < 500 && "Destructor took too long (> 500ms)");

  std::cout << "[test] testCleanFastDestruction PASSED\n";
}

int main() {
  std::cout << "=== Running Data Inspector Native Tests ===\n";

  testShmServerBasics();
  testInspectorAppLifecycleAndAovs();
  testBundleManifestLoading();
  testCleanFastDestruction();

  std::cout << "=== All Data Inspector Native Tests PASSED ===\n";
  return 0;
}
