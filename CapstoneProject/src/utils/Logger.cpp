// hide logger's implementation details

#include "logger.hpp"
#include <chrono>
#include <cstdlib>
#include <string_view>

namespace {
  const auto g_t0 = std::chrono::steady_clock::now();
}

namespace logger {
  thread_local const char* tlabel = "main";

  uint64_t ms_since_start() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_t0).count());
  }

  bool verbose() {
    const char* v = std::getenv("VERBOSE");
    return v && *v && std::string_view(v) != "0";
  }
}