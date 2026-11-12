/// @file main.cc
/// @brief Entry point for the Data Inspector native rendering subprocess.
///
/// Usage:
///   data-inspector-native [options]
///
/// Options:
///   --rhi=<backend>       RHI backend: vulkan|metal|d3d12|null (default: null)
///   --headless            Run without a visible window (default: true unless --windowed)
///   --windowed            Run in windowed mode
///   --width=<W>           Frame width  (default: 1280)
///   --height=<H>          Frame height (default: 720)
///   --frames=<N>          Exit after rendering N frames (default: 0 = continuous)
///   --shm-name=<name>     SHM region name (default: /inspector-frame)
///   --keep-shm            Do not unlink SHM region on exit (for CLI/test inspection)
///   --bundle=<path>       Path to the InspectionBundle directory

#include "inspector_app.h"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

[[nodiscard]] std::string getArg(int argc, char** argv, std::string_view key,
                                 std::string_view fallback = "") {
  const std::string prefix = "--" + std::string(key) + "=";
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.substr(0, prefix.size()) == prefix) {
      return std::string(arg.substr(prefix.size()));
    }
  }
  return std::string(fallback);
}

[[nodiscard]] bool hasFlag(int argc, char** argv, std::string_view flag) {
  const std::string full = "--" + std::string(flag);
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == full) return true;
  }
  return false;
}

[[nodiscard]] bool parseUint32(std::string_view text, uint32_t& out) noexcept {
  if (text.empty()) return false;
  uint32_t val = 0;
  const char* begin = text.data();
  const char* end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, val);
  if (ec != std::errc{} || ptr != end) {
    return false;
  }
  out = val;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  using synapse::inspector::InspectorApp;
  using synapse::inspector::InspectorConfig;
  using synapse::inspector::kDefaultShmName;

  InspectorConfig config;
  config.rhiBackend = getArg(argc, argv, "rhi", "null");
  config.headless = hasFlag(argc, argv, "headless") ||
                    !hasFlag(argc, argv, "windowed");
  config.keepShm = hasFlag(argc, argv, "keep-shm");
  config.shmName =
      getArg(argc, argv, "shm-name", std::string(kDefaultShmName));
  config.bundlePath = getArg(argc, argv, "bundle", "");

  const std::string wStr = getArg(argc, argv, "width", "1280");
  const std::string hStr = getArg(argc, argv, "height", "720");
  const std::string framesStr = getArg(argc, argv, "frames", "0");

  if (!parseUint32(wStr, config.width) || !parseUint32(hStr, config.height)) {
    std::cerr << "[inspector] Invalid width/height: " << wStr << "x" << hStr
              << "\n";
    return 1;
  }

  if (!parseUint32(framesStr, config.maxFrames)) {
    std::cerr << "[inspector] Invalid frames count: " << framesStr << "\n";
    return 1;
  }

  if (config.width == 0 || config.height == 0 || config.width > 7680 ||
      config.height > 4320) {
    std::cerr << "[inspector] Resolution out of range: " << config.width << "x"
              << config.height << "\n";
    return 1;
  }

  std::cout << "[inspector] Data Inspector Native v0.1.0\n";

  InspectorApp app(std::move(config));

  if (!app.init()) {
    std::cerr << "[inspector] Initialisation failed\n";
    return 1;
  }

  app.run();

  std::cout << "[inspector] Clean exit\n";
  return 0;
}
