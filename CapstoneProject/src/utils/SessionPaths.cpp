#include "SessionPaths.hpp"

// Find the project root by walking upward from current working directory.
// "project root" is the directory that contains both:
//   - data/  (directory)
//   - models/ (directory)
// If not found within max_depth, returns current_path() as fallback
fs::path sesspaths::find_project_root(int max_depth) {
    fs::path p = fs::current_path();
    SESS_LOG("find_project_root: cwd=" << p.string());

    for (int i = 0; i < max_depth; ++i) {
        const fs::path data_dir   = p / "data";
        const fs::path models_dir = p / "models";

        const bool has_data   = fs::exists(data_dir)   && fs::is_directory(data_dir);
        const bool has_models = fs::exists(models_dir) && fs::is_directory(models_dir);

        if (has_data && has_models) {
            SESS_LOG("find_project_root: FOUND root=" << p.string());
            return p;
        }

        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }

    fs::path fallback = fs::current_path();
    SESS_LOG("find_project_root: NOT FOUND (max_depth=" << max_depth
        << "), fallback=" << fallback.string());
    return fallback;
}


// Allocate a fallback subject id: person1, person2, ...
// Persist the counter in: <root>/data/.next_person_id
std::string sesspaths::allocate_person_fallback(const fs::path& data_root_dir) {
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

// delete all sessions for current user execpt the <keep_n> most recent ones 
void sesspaths::prune_old_sessions_for_subject(const fs::path& subject_dir, std::size_t keep_n) {
    std::error_code ec;

    if (!fs::exists(subject_dir, ec) || !fs::is_directory(subject_dir, ec)) return;

    struct Entry {
        fs::path path;
        fs::file_time_type t;
    };

    std::vector<Entry> sessions;
    for (const auto& de : fs::directory_iterator(subject_dir, ec)) {
        if (ec) break;
        if (!de.is_directory(ec)) continue;

        const std::string name = de.path().filename().string();
        if (!is_session_dir_name(name)) continue;

        fs::file_time_type t = de.last_write_time(ec);
        if (ec) t = fs::file_time_type::min(); // fallback
        sessions.push_back({de.path(), t});
    }

    if (sessions.size() <= keep_n) return;

    // newest first
    std::sort(sessions.begin(), sessions.end(),
              [](const Entry& a, const Entry& b) { return a.t > b.t; });

    for (std::size_t i = keep_n; i < sessions.size(); ++i) {
        std::error_code rmec;
        LOG_ALWAYS("prune: removing old session dir " << sessions[i].path.string());
        fs::remove_all(sessions[i].path, rmec);
        if (rmec) {
            LOG_ALWAYS("prune: ERROR removing " << sessions[i].path.string()
                      << " (" << rmec.message() << ")");
        }
    }
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
SessionPaths sesspaths::create_session(const std::string& preferred_subject_name) {
    SessionPaths sp{};

    SESS_LOG("create_session: BEGIN preferred_subject_name='" << preferred_subject_name << "'");

    // 1) Find project root (CapstoneProject)
    sp.project_root = find_project_root();
    SESS_LOG("create_session: project_root=" << sp.project_root.string());

    const fs::path data_root   = sp.project_root / "data";
    const fs::path models_root = sp.project_root / "models";

    std::error_code ec;
    fs::create_directories(data_root, ec);
    ec.clear();
    fs::create_directories(models_root, ec);
    ec.clear();

    // 2) Determine subject id
    std::string preferred = preferred_subject_name;
    trim_in_place(preferred);

    bool used_fallback = false;
    std::string subj;

    if (preferred.empty()) {
        // User provided no meaningful name (spaces), so we fallback to personN
        used_fallback = true;
        subj = allocate_person_fallback(data_root);
    } else {
        // User provided something, so we sanitize and accept
        subj = sanitize_subject_id(preferred);
        // Another policy: Require at least one letter, otherwise fallback
        if (!contains_alpha(subj)) {
            used_fallback = true;
            subj = allocate_person_fallback(data_root);
        }
    }

    sp.subject_id = subj;
    SESS_LOG("create_session: subject_id=" << sp.subject_id
        << " (fallback=" << (used_fallback ? "Y" : "N") << ")");

    // 3) Create session id
    sp.session_id = make_session_id_timestamp();
    SESS_LOG("create_session: session_id=" << sp.session_id);

    // 4) Build session dirs + create them
    sp.data_session_dir  = data_root / sp.subject_id / sp.session_id;
    sp.model_session_dir = models_root / sp.subject_id / sp.session_id;

    SESS_LOG("create_session: data_session_dir=" << sp.data_session_dir.string());
    SESS_LOG("create_session: model_session_dir=" << sp.model_session_dir.string());

    fs::create_directories(sp.data_session_dir, ec);
    SESS_LOG("create_session: create_directories data_session_dir -> " << ec_str(ec));
    ec.clear();

    fs::create_directories(sp.model_session_dir, ec);
    SESS_LOG("create_session: create_directories model_session_dir -> " << ec_str(ec));
    ec.clear();

    // verify they exist
    const bool data_ok  = fs::exists(sp.data_session_dir)  && fs::is_directory(sp.data_session_dir);
    const bool model_ok = fs::exists(sp.model_session_dir) && fs::is_directory(sp.model_session_dir);

    if (!data_ok || !model_ok) {
        SESS_LOG("create_session: WARNING session dirs missing after create_directories");
        // todo: decide if u want hard fail
        // throw std::runtime_error("create_session: failed to create session directories");
    }

    // After creating new dirs
    prune_old_sessions_for_subject(data_root / sp.subject_id, 3);
    prune_old_sessions_for_subject(models_root / sp.subject_id, 3);

    return sp;
}


