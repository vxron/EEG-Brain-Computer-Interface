// utils/SessionPaths.hpp
// -----------------------------------------------------------------------------
// Session folder infrastructure
//
// Goal:
//   - Always write outputs under:
//       <Capstine>/data/<subject_id>/<session_id>/...
//       <Capstone>/models/<subject_id>/<session_id>/...
//   (even when executable is launched from out/build/... )
//   - Use subject name (from StateStore) if available; else fallback
//     to person1, person2, ... 
//   - Note this is header-only for simplicity.
//   - New session should be created (with these new dirs) everytime we enter calib mode
// -----------------------------------------------------------------------------

#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cctype>
#include <algorithm>
#include <vector>
#include <system_error>
#include "Types.h"
#include "Logger.hpp"

#define SESS_LOG(msg) LOG_ALWAYS("sesspaths: " << msg)
namespace sesspaths {

namespace fs = std::filesystem;

static constexpr const char* kInProgressSuffix = "__IN_PROGRESS";

//  Inline Helpers
static inline std::string ec_str(const std::error_code& ec) {
    if (!ec) return "ok";
    std::ostringstream oss;
    oss << ec.value() << " (" << ec.category().name() << "): " << ec.message();
    return oss.str();
}

static inline bool is_session_dir_name(const std::string& name) {
    // session dirs are timestamps like "2025-12-28_16-54-25"
    return !name.empty() && std::isdigit((unsigned char)name[0]);
}

static inline void trim_in_place(std::string& s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

    while (!s.empty() && is_space(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());

    while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))
        s.pop_back();
}

static inline bool contains_alpha(const std::string& s) {
    for (unsigned char c : s) {
        if (std::isalpha(c)) return true;
    }
    return false;
}

// Allowed: [A-Za-z0-9_-]. Everything else becomes '_'.
static inline std::string sanitize_subject_id(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))  s.pop_back();

    for (char& c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        const bool ok = (std::isalnum(uc) != 0) || (c == '_') || (c == '-');
        if (!ok) c = '_';
    }

    if (s.empty()) s = "unknown";
    return s;
}

// Generate a session id like: 2025-12-22_14-31-08
static inline std::string make_session_id_timestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

static inline fs::path data_file(const SessionPaths& sp, const std::string& filename) {
    return sp.data_session_dir / filename;
}

static inline fs::path model_file(const SessionPaths& sp, const std::string& filename) {
    return sp.model_session_dir / filename;
}

static inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static inline std::string with_in_progress_suffix(const std::string& session_id_base) {
    return session_id_base + kInProgressSuffix;
}

static inline bool is_in_progress_session_id(const std::string& session_id) {
    return ends_with(session_id, kInProgressSuffix);
}

static inline std::string strip_in_progress_suffix(std::string session_id) {
    const std::string suf = kInProgressSuffix;
    if (ends_with(session_id, suf)) {
        session_id.resize(session_id.size() - suf.size());
    }
    return session_id;
}

// declarations
fs::path find_project_root(int max_depth = 12);
std::string allocate_person_fallback(const fs::path& data_root_dir);
void prune_old_sessions_for_subject(const fs::path& subject_dir, std::size_t keep_n);
SessionPaths create_session(const std::string& preferred_subject_name);
void delete_session_dirs_if_in_progress(const SessionPaths& sp);
bool finalize_session_dirs(SessionPaths& sp); 

} // namespace sesspaths