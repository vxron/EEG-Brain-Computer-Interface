#include "logger.hpp"
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <iostream>
#include <mutex>

namespace {
  const auto g_t0 = std::chrono::steady_clock::now();
  std::mutex g_log_mtx;
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

  std::mutex& log_mutex() {
    return g_log_mtx;
  }

  void write_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(g_log_mtx);
    std::cout << line << std::endl;
  }
}
