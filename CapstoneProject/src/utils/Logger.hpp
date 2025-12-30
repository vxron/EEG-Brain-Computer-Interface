#pragma once
#include <cstdint>
#include <string>
#include <mutex>

namespace logger {
  // Returns ms since program start (steady clock).
  uint64_t ms_since_start();

  // True if VERBOSE env var is set and not "0".
  bool verbose();

  // Per-thread label used in log lines (defaults to "main").
  extern thread_local const char* tlabel;

  // Global log mutex (single sink).
  std::mutex& log_mutex();

  // Thread-safe write of a fully formatted line.
  void write_line(const std::string& line);
}

// Macros build the full line, then print under a mutex.
#define LOG_ALWAYS(msg) do { \
  std::ostringstream _oss; \
  _oss << "[" << std::setw(6) << logger::ms_since_start() << " ms] " \
       << logger::tlabel << ": " << msg; \
  logger::write_line(_oss.str()); \
} while(0)

#define LOG_DBG(msg) do { if (logger::verbose()) LOG_ALWAYS(msg); } while(0)
