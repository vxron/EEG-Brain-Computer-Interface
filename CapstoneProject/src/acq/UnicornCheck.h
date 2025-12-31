// unicorn_check.h
#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include "../utils/Logger.hpp"
#include <vector>
#include <iomanip>

#include "../../unicorn/include/unicorn.h"

inline constexpr int UNICORN_SAMPLING_RATE_HZ = UNICORN_SAMPLING_RATE;

// Simple exception type for hard failures
struct unicorn_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Implementation helpers 
inline void uc_check_impl(int ec,
                          const char* where,
                          const char* file,
                          int line,
                          const char* func)
{
    if (ec == UNICORN_ERROR_SUCCESS) return;
    const char* msg = UNICORN_GetLastErrorText(); // capture immediately
    std::ostringstream oss;
    oss << where << " failed at " << file << ":" << line
        << " in " << func << " -> " << (msg ? msg : "(no error text)");
    throw unicorn_error(oss.str());
}

inline bool uc_warn_if_fail_impl(int ec,
                                 const char* where,
                                 const char* file,
                                 int line,
                                 const char* /*func*/)
{
    if (ec == UNICORN_ERROR_SUCCESS) return true;

    const char* msg = UNICORN_GetLastErrorText();
    logger::tlabel = "Unicorn";

    std::ostringstream oss;
    oss << where << " failed at " << file << ":" << line
        << " -> " << (msg ? msg : "(no error text)");
    LOG_ALWAYS(oss.str());

    return false;
}

inline void uc_log_result_impl(int ec,
                               const char* where,
                               const char* file,
                               int line)
{
    logger::tlabel = "Unicorn";
    const char* msg = UNICORN_GetLastErrorText();

    std::ostringstream oss;
    if (ec == UNICORN_ERROR_SUCCESS) {
        oss << where << ": OK (" << file << ":" << line << ")";
    } else {
        oss << where << ": " << (msg ? msg : "(no error text)")
            << " (" << file << ":" << line << ")";
    }
    LOG_ALWAYS(oss.str());
}

// Caller-friendly wrappers/macros 
#define UCHECK(expr) \
    ::uc_check_impl((expr), #expr, __FILE__, __LINE__, __func__)

#define UWARN_IF_FAIL(expr) \
    ::uc_warn_if_fail_impl((expr), #expr, __FILE__, __LINE__, __func__)

#define ULOG_RESULT(expr) \
    ::uc_log_result_impl((expr), #expr, __FILE__, __LINE__)
