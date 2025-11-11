#pragma once
#include <cstdint>
#include <iostream>
#include <iomanip>

namespace logger {
  // Returns ms since program start (steady clock).
  uint64_t ms_since_start();

  // True if VERBOSE env var is set and not "0".
  bool verbose();

  // Per-thread label used in log lines (defaults to "main").
  extern thread_local const char* tlabel;
}

// Pretty simple macros. Keep I/O headers included above.
#define LOG_ALWAYS(msg) do { \
  std::cout << "[" << std::setw(6) << logger::ms_since_start() << " ms] " \
            << logger::tlabel << ": " << msg << std::endl; \
} while(0)

#define LOG_DBG(msg) do { if (logger::verbose()) LOG_ALWAYS(msg); } while(0)