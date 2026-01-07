

// httplib::request = everything that comes from the client (HTML/JS browser)
// httplib::responde = everything that server writes back to client
// data exchanges are in json format

#include "HttpServer.hpp"
#include <atomic>
#include <string>
#include <sstream>
#include <iomanip>
#include <cctype>
#include "../utils/JsonUtils.hpp"

// Constructor
HttpServer_C::HttpServer_C(StateStore_s& stateStoreRef, int port) : stateStoreRef_(stateStoreRef), liveServerRef_(nullptr), port_(port) {
}

// Destructor 
HttpServer_C::~HttpServer_C() {
    // need to unallocate dynamically allocated ptr ref (done w 'new')
    delete liveServerRef_;
}

// ============= Helpers ============
static inline void set_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

// Writes JSON string into httplib:response body with correct CORS header
void HttpServer_C::write_json(httplib::Response& res, std::string_view json_body) const {
    set_cors_headers(res);
    res.set_content(json_body.data(), "application/json");
    res.status = 200;
}

// Allow methods/headers letting UI make POST requests
void HttpServer_C::handle_options_and_set(const httplib::Request& req, httplib::Response& res) {
    set_cors_headers(res);
    res.status = 200;
}

static inline void write_json_error(httplib::Response& res,
                                    int status,
                                    const char* error,
                                    const char* field = nullptr)
{
    set_cors_headers(res);
    res.status = status;

    std::ostringstream oss;
    oss << "{"
        << "\"ok\":false,"
        << "\"error\":\"" << error << "\"";
    if (field) oss << ",\"field\":\"" << field << "\"";
    oss << "}";

    res.set_content(oss.str(), "application/json");
}


// ============== Handlers ==================

void HttpServer_C::handle_get_state(const httplib::Request& req, httplib::Response& res){
    /* Goal: (upon client polling request)
    - read a snapshot of current statestore_s 
    - turn it into a little json string
    - send back the json with res */ 
    (void)req; //unused

    // 1) "snapshot" read of current statestore_s
    int seq = stateStoreRef_.g_ui_seq.load(std::memory_order_acquire); // stimcontroller is only one who touches this
    int stim_window = static_cast<int>(stateStoreRef_.g_ui_state.load(std::memory_order_acquire));

    // Calib freqs (training protocol)
    int block_id = stateStoreRef_.g_block_id.load(std::memory_order_acquire);
    int freq_hz_e = static_cast<int>(stateStoreRef_.g_freq_hz_e.load(std::memory_order_acquire));
    int freq_hz = stateStoreRef_.g_freq_hz.load(std::memory_order_acquire);
    
    // Run-mode pair
    std::lock_guard<std::mutex> lock(stateStoreRef_.saved_sessions_mutex);
    int currIdx = stateStoreRef_.currentSessionIdx.load(std::memory_order_acquire);
    int n = static_cast<int>(stateStoreRef_.saved_sessions.size());
    // guard against out of range errors
    if(currIdx < 0){currIdx = 0;}
    if(currIdx >= n){currIdx = n-1;}
    int freq_left_hz   = stateStoreRef_.saved_sessions[currIdx].freq_left_hz;
    int freq_right_hz  = stateStoreRef_.saved_sessions[currIdx].freq_right_hz;
    int freq_left_hz_e = static_cast<int>(stateStoreRef_.saved_sessions[currIdx].freq_left_hz_e);
    int freq_right_hz_e = static_cast<int>(stateStoreRef_.saved_sessions[currIdx].freq_right_hz_e);

    // Active session info
    bool is_model_ready = stateStoreRef_.currentSessionInfo.g_isModelReady.load(std::memory_order_acquire); // for training job monitoring
    std::string active_subject_id = stateStoreRef_.currentSessionInfo.get_active_subject_id();
    int popup = stateStoreRef_.g_ui_popup.load(std::memory_order_acquire); // any popup event

    // Pending session info
    std::lock_guard<std::mutex> lock2(stateStoreRef_.calib_options_mtx);
    std::string pending_subject_name = stateStoreRef_.pending_subject_name;

    // Settings so JS renders correct toggle on entry
    int calib_data_setting_e = stateStoreRef_.settings.calib_data_setting.load(std::memory_order_acquire);
    int train_arch_e = stateStoreRef_.settings.train_arch_setting.load(std::memory_order_acquire);

    // 2) build json string manually
    std::ostringstream oss;
    oss << "{"
        << "\"seq\":"                    << seq                                 << ","
        << "\"stim_window\":"            << stim_window                         << ","
        << "\"block_id\":"               << block_id                            << ","
        << "\"freq_hz\":"                << freq_hz                             << ","
        << "\"freq_hz_e\":"              << freq_hz_e                           << ","
        << "\"freq_left_hz\":"           << freq_left_hz                        << ","
        << "\"freq_right_hz\":"          << freq_right_hz                       << ","
        << "\"freq_left_hz_e\":"         << freq_left_hz_e                      << ","
        << "\"freq_right_hz_e\":"        << freq_right_hz_e                     << ","
        << "\"is_model_ready\":"         << (is_model_ready ? "true" : "false") << ","
        << "\"popup\":"                  << popup                               << ","
        << "\"pending_subject_name\":\"" << pending_subject_name                << "\","
        << "\"active_subject_id\":\""    << active_subject_id                   << "\","
        << "\"settings\":{"
            << "\"calib_data_setting\":" << calib_data_setting_e << ","
            << "\"train_arch_setting\":" << train_arch_e
        << "}"
        << "}";

    std::string json_snapshot = oss.str();
    
    // 3) send json back to client through res
    write_json(res, json_snapshot);
}

// handle ui state transition events written by JS
void HttpServer_C::handle_post_event(const httplib::Request& req, httplib::Response& res){
    // Basic content-type check
    auto it = req.headers.find("Content-Type");
    if (it == req.headers.end() || it->second.find("application/json") == std::string::npos) {
        JSON::json_extract_fail("http_server", "Content-Type");
        write_json_error(res, 415, "unsupported_media_type", "Content-Type");
        return;
    }
    
    const std::string& body = req.body;
    UIStateEvent_E ev = UIStateEvent_None;

    auto p = body.find("\"action\"");
    if (p == std::string::npos) {
        JSON::json_extract_fail("http_server", "action");
        write_json_error(res, 400, "missing_or_invalid_field", "action");
        return;
    }
    if (p != std::string::npos) {
        p = body.find(':', p);
        if (p != std::string::npos) {
            p = body.find('"', p);       // opening quote
            auto q = body.find('"', p+1); // closing quote
            if (p != std::string::npos && q != std::string::npos && q > p+1) {
                std::string action = body.substr(p+1, q - (p+1));
                if (action == "start_calib") {
                    ev = UIStateEvent_UserPushesStartCalib;
                } else if (action == "start_run") {
                    ev = UIStateEvent_UserPushesStartRun;
                } else if (action == "exit"){
                    ev = UIStateEvent_UserPushesExit;
                } else if (action == "start_default"){
                    ev = UIStateEvent_UserPushesStartDefault;
                } else if (action == "show_sessions"){
                    ev = UIStateEvent_UserPushesSessions;
                } else if (action == "new_session"){
                    ev = UIStateEvent_UserSelectsNewSession;
                } else if (action == "back_to_run_options"){
                    ev = UIStateEvent_UserPushesStartRun;
                } 
                else if (action == "ack_popup"){
                    // clear popup when user presses OK
                    ev = UIStateEvent_UserAcksPopup;
                    stateStoreRef_.g_ui_popup.store(UIPopup_None, std::memory_order_release);
                } 
                else if (action == "cancel_popup") {
                    // clear popup (and event should be swallowed)
                    ev = UIStateEvent_UserCancelsPopup;
                    stateStoreRef_.g_ui_popup.store(UIPopup_None, std::memory_order_release);
                } 
                else if (action == "hardware_checks"){
                    ev = UIStateEvent_UserPushesHardwareChecks;
                } 
                else if (action == "start_calib_from_options") {
                    
                    // read form fields from JSON
                    std::string subj;
                    int epilepsy_i = static_cast<int>(EpilepsyRisk_Unknown);
                    if (!(JSON::extract_json_string(body, "\"subject_name\"", subj))) {
                        JSON::json_extract_fail("http_server", "subject_name");
                        write_json_error(res, 400, "missing_or_invalid_field", "subject_name");
                        return;
                    }
                    if(!(JSON::extract_json_int(body, "\"epilepsy\"", epilepsy_i))){
                        JSON::json_extract_fail("http_server", "epilepsy");
                        write_json_error(res, 400, "missing_or_invalid_field", "epilepsy");
                        return;
                    };
                    // publish to statestore
                    {
                        std::lock_guard<std::mutex> lock(stateStoreRef_.calib_options_mtx);
                        stateStoreRef_.pending_subject_name = subj;
                        stateStoreRef_.pending_epilepsy = static_cast<EpilepsyRisk_E>(epilepsy_i);
                    }
                    ev = UIStateEvent_UserPushesStartCalibFromOptions;
                } 
                else if (action == "open_settings") {
                    ev = UIStateEvent_UserPushesSettings;
                }
                else if (action == "set_settings") {
                    int setting_i = static_cast<int>(CalibData_MostRecentOnly); // default
                    if (!JSON::extract_json_int(body, "\"calib_data_setting\"", setting_i)) {
                        JSON::json_extract_fail("http_server", "calib_data_setting");
                        write_json_error(res, 400, "missing_or_invalid_field", "calib_data_setting");
                        return;
                    }
                    stateStoreRef_.settings.calib_data_setting.store(static_cast<SettingCalibData_E>(setting_i), std::memory_order_release);
                    
                    int arch_i = static_cast<int>(TrainArch_CNN); // default
                    if (!JSON::extract_json_int(body, "\"train_arch_setting\"", arch_i)) {
                        JSON::json_extract_fail("http_server", "train_arch_setting");
                        write_json_error(res, 400, "missing_or_invalid_field", "train_arch");
                        return;
                    }
                    stateStoreRef_.settings.train_arch_setting.store(static_cast<SettingTrainArch_E>(arch_i), std::memory_order_release);
                }
                else {
                    // Unknown action (error)
                    JSON::json_extract_fail("http_server", action.c_str());
                    write_json_error(res, 400, "unknown_action", action.c_str());
                    return;
                }
            }
        }
    }
    // only stim controller should be able to write None to shared state when it's finished processing an event
    if (ev != UIStateEvent_None) {
        stateStoreRef_.g_ui_event.store(ev, std::memory_order_release);
    }

    write_json(res, "{\"ok\":true}");
}

// returns rolling stats 
// TODO: Add pipeline for 'quality' assessment (red/yellow/green) BASED ON CURRENT BAD WINDOW RATE
// ^^maybe implement in js honestly
void HttpServer_C::handle_get_quality(const httplib::Request& req,
                                      httplib::Response& res)
{
    (void)req;

    int n_ch = stateStoreRef_.g_n_eeg_channels.load(std::memory_order_acquire);
    if (n_ch <= 0 || n_ch > NUM_CH_CHUNK) n_ch = NUM_CH_CHUNK; // default to the 8 channels

    // Get rolling stats snapshot (mtx-protected in statestore)
    SignalStats_s ss = stateStoreRef_.get_signal_stats();
    const Stats_s rollingStats = ss.rollingStats;
    
    std::ostringstream oss;
    oss << "{";
    oss << "\"n_channels\":" << n_ch << ",";

    // labels
    oss << "\"labels\":[";
    for (int ch = 0; ch < n_ch; ++ch) {
        const std::string& lbl = stateStoreRef_.eeg_channel_labels[ch];
        oss << "\"" << lbl << "\"";
        if (ch < n_ch - 1) oss << ",";
    }
    oss << "],";

    // rolling stats arrays - lambda helper :)
    auto write_arr = [&](const char* key, const auto& arr){
        oss << "\"" << key << "\":[";
        for (int ch = 0; ch < n_ch; ++ch) {
            oss << arr[ch];
            if (ch < n_ch - 1) oss << ",";
        }
        oss << "]";
    };

    oss << "\"rolling\":{";
    write_arr("mean_uv",     rollingStats.mean_uv);     oss << ",";
    write_arr("std_uv",      rollingStats.std_uv);      oss << ",";
    write_arr("rms_uv",      rollingStats.rms_uv);      oss << ",";
    write_arr("mad_uv",      rollingStats.mad_uv);      oss << ",";
    write_arr("max_abs_uv",  rollingStats.max_abs_uv);  oss << ",";
    write_arr("max_step_uv", rollingStats.max_step_uv); oss << ",";
    write_arr("kurt",        rollingStats.kurt);        oss << ",";
    write_arr("entropy",     rollingStats.entropy);
    oss << "},";

    // summary rates (rolling + overall)
    oss << "\"rates\":{"
        << "\"current_bad_win_rate\":" << ss.current_bad_win_rate << ","
        << "\"overall_bad_win_rate\":" << ss.overall_bad_win_rate << ","
        << "\"num_win_in_rolling\":"   << ss.num_win_in_rolling
        << "}";

    oss << "}";
    write_json(res, oss.str());
}

// writes eeg channel configs, labels + actual eeg samples for plotting on UI
void HttpServer_C::handle_get_eeg(const httplib::Request& req,
                                  httplib::Response& res)
{
    (void)req;

    if (!stateStoreRef_.g_hasEegChunk.load(std::memory_order_acquire)) {
        write_json(res, "{\"ok\":false,\"msg\":\"no eeg yet\"}");
        return;
    }

    bufferChunk_S last = stateStoreRef_.get_lastEegChunk();

    int n_ch = stateStoreRef_.g_n_eeg_channels.load(std::memory_order_acquire);
    if (n_ch <= 0 || n_ch > NUM_CH_CHUNK) {
        n_ch = NUM_CH_CHUNK; // fallback
    }

    const int stride = NUM_CH_CHUNK; // interleave stride in bufferChunk_S
    const int samples_per_channel = NUM_SAMPLES_CHUNK / stride;

    std::ostringstream oss;
    oss << "{"
        << "\"ok\":true,"
        << "\"fs\":250,"
        << "\"units\":\"uV\","
        << "\"n_channels\":" << n_ch << ",";

    // labels
    oss << "\"labels\":[";
    for (int ch = 0; ch < n_ch; ++ch) {
        const std::string& lbl = stateStoreRef_.eeg_channel_labels[ch];
        oss << "\"" << lbl << "\"";
        if (ch < n_ch - 1) oss << ",";
    }
    oss << "],";

    // channels
    oss << "\"channels\":[";
    for (int ch = 0; ch < n_ch; ++ch) {
        oss << "[";
        for (int s = 0; s < samples_per_channel; ++s) {
            int idx = s * stride + ch;  // time-major interleave
            oss << last.data[idx];
            if (s < samples_per_channel - 1) oss << ",";
        }
        oss << "]";
        if (ch < n_ch - 1) oss << ",";
    }

    oss << "]}";
    write_json(res, oss.str());
}


// check ready and write monitor refresh rate from client response
void HttpServer_C::handle_post_ready(const httplib::Request& req, httplib::Response& res){
    /* 
    - HTML/JS is the one that measures acc monitor refresh rate (with requestAnimationFrame)
    - JS sends results as a POST req body
    - handle_post_ready must read req.body and parse out refresh_hz
    */
   int hz = 0;
   const std::string& body = req.body; // 1) read the JSON string from browser containing POST with refresh_hz
   auto p = body.find("\"refresh_hz\""); // 2) find the substring "refresh_hz"
   if(p!=std::string::npos){
    p = body.find(':',p); // 3) find the colon after "refresh_hz" to find acc value
    if(p!=std::string::npos){
        ++p;
        while(p<body.size() && body[p]==' '){
            ++p; // 4) skip spaces
        }
        while(p<body.size() && isdigit((unsigned char)body[p])){
            hz=hz*10+(body[p]-'0'); // 5) read refresh_hz and cast to int from hz op
            ++p;
        }
    }
   }
   if(hz > 0){
    // 6) make sure val is positive n makes sense, then store
    stateStoreRef_.g_refresh_hz.store(hz, std::memory_order_release); // use 'release' for writers
   }
   write_json(res, "{\"ok\":true}"); // 7) small msg back to browser to say all good
}

// ===================== Lifecycle ==========================
bool HttpServer_C::http_start_server(){
    logger::tlabel = "HTTP Server";
    if (is_running_.load()) return false;

    liveServerRef_ = new httplib::Server(); // listens for requests from HTML/JS

    liveServerRef_->set_default_headers({
    // no Access-Control-* here; CORS handled per-response
    });

    // Route bindings
    liveServerRef_->Get("/state",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_get_state(rq, rs); });

    liveServerRef_->Get("/quality",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_get_quality(rq, rs); });

    liveServerRef_->Get("/eeg",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_get_eeg(rq, rs); });

    liveServerRef_->Post("/event",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_post_event(rq, rs); });

    liveServerRef_->Post("/ready",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_post_ready(rq, rs); });

    // CORS preflight for POSTs
    liveServerRef_->Options("/event",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_options_and_set(rq, rs); });

    liveServerRef_->Options("/ready",
        [this](const httplib::Request& rq, httplib::Response& rs){ this->handle_options_and_set(rq, rs); });

    LOG_ALWAYS("HTTP Server successfully opened");
    return true;
}

bool HttpServer_C::http_listen_for_poll_requests(){
    /* goal:
    - blocking function to call from inside server thread in main (main lifecycle of server)
    */
   // 1) check that server object exits
   logger::tlabel = "HTTP Server";
   if(liveServerRef_ == nullptr) {
    LOG_ALWAYS("HTTP server not initialized; cannot start listening");
    return false;
   }
   is_running_.store(true, std::memory_order_release);
   LOG_ALWAYS("HTTP listening on 127.0.0.1: " << port_);

   // 2) start listening to browser (blocking)
   bool ok = liveServerRef_->listen("127.0.0.1", port_);
   // 3) handle listen() returning & log
   is_running_.store(false, std::memory_order_release);

   if(!ok){
    LOG_ALWAYS("HTTP listen failed on port " << port_);
   } else {
    LOG_ALWAYS("HTTP listen stopped successfully");
   }
   return ok;
}

bool HttpServer_C::http_close_server(){
    logger::tlabel = "HTTP Server";
    if (!liveServerRef_) return false;
    liveServerRef_->stop(); // breaks .listen()
    LOG_ALWAYS("HTTP Server successfully closed");
    return true;
}




