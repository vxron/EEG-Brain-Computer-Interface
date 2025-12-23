// utils/SessionPaths.hpp
// -----------------------------------------------------------------------------
// Session folder infrastructure
//
// Goal:
//   - Always write outputs under:
//       <CapstoneProject>/data/<subject_id>/<session_id>/...
//       <CapstoneProject>/models/<subject_id>/<session_id>/...
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
#include "Types.h"

namespace capstone {
namespace sesspaths {

namespace fs = std::filesystem;

//  Inline Helpers

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

// Find the project root by walking upward from current working directory.
// "project root" is the directory that contains both:
//   - data/  (directory)
//   - models/ (directory)
// If not found within max_depth, returns current_path() as fallback
static inline fs::path find_project_root(int max_depth = 12) {
    fs::path p = fs::current_path();

    for (int i = 0; i < max_depth; ++i) {
        const fs::path data_dir   = p / "data";
        const fs::path models_dir = p / "models";

        if (fs::exists(data_dir) && fs::is_directory(data_dir) &&
            fs::exists(models_dir) && fs::is_directory(models_dir)) {
            return p; // Found CapstoneProject root
        }

        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    return fs::current_path();
}

// Allocate a fallback subject id: person1, person2, ...
// Persist the counter in: <root>/data/.next_person_id
static inline std::string allocate_person_fallback(const fs::path& data_root_dir) {
    // Ensure data root exists so we can store the counter file
    std::error_code ec;
    fs::create_directories(data_root_dir, ec);

    const fs::path counter_path = data_root_dir / ".next_person_id";

    int next_id = 1;

    // Read counter if present
    {
        std::ifstream in(counter_path.string());
        if (in.good()) {
            in >> next_id;
            if (!in.good()) next_id = 1;
        }
    }

    if (next_id < 1) next_id = 1;

    // Write incremented counter back
    {
        std::ofstream out(counter_path.string(), std::ios::trunc);
        if (out.good()) {
            out << (next_id + 1);
        }
        // If this fails, still return personN; it just won't persist
    }

    return "person" + std::to_string(next_id);
}

// PUBLIC API
// Create a new session.
// Inputs:
//   preferred_subject_name:
//     - pass user's chosen name (from StateStore) if available
//     - pass "" if fallback to personN
//
// Behavior:
//   - Finds project root 
//   - Creates:
//      <root>/data/<subject>/<session>/
//      <root>/models/<subject>/<session>/
//   - Returns paths + ids.
SessionPaths create_session(const std::string& preferred_subject_name) {
    SessionPaths sp{};

    // 1) Find project root (CapstoneProject)
    sp.project_root = find_project_root();

    const fs::path data_root   = sp.project_root / "data";
    const fs::path models_root = sp.project_root / "models";

    std::error_code ec;
    fs::create_directories(data_root, ec);
    fs::create_directories(models_root, ec);

    // 2) Determine subject id
    std::string subj = preferred_subject_name;
    subj = sanitize_subject_id(subj);

    // If nothing meaningful came in, use personN allocator.
    if (preferred_subject_name.empty() || subj == "unknown") {
        subj = allocate_person_fallback(data_root);
    }

    sp.subject_id = subj;

    // 3) Create session id
    sp.session_id = make_session_id_timestamp();

    // 4) Build session dirs + create them
    sp.data_session_dir  = data_root / sp.subject_id / sp.session_id;
    sp.model_session_dir = models_root / sp.subject_id / sp.session_id;

    fs::create_directories(sp.data_session_dir, ec);
    fs::create_directories(sp.model_session_dir, ec);

    return sp;
}

static inline fs::path data_file(const SessionPaths& sp, const std::string& filename) {
    return sp.data_session_dir / filename;
}

static inline fs::path model_file(const SessionPaths& sp, const std::string& filename) {
    return sp.model_session_dir / filename;
}

} // namespace sesspaths
} // namespace capstone
