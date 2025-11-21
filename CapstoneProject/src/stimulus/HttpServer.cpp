

// httplib::request = everything that comes from the client (HTML/JS browser)
// httplib::responde = everything that server writes back to client
// data exchanges are in json format

#include "HttpServer.hpp"
#include <atomic>
#include <string>
#include <sstream>
#include <iomanip>
#include <cctype>

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

// ============== Handlers ==================

void HttpServer_C::handle_get_state(const httplib::Request& req, httplib::Response& res){
    /* Goal: (upon client polling request)
    - read a snapshot of current statestore_s 
    - turn it into a little json string
    - send back the json with res */ 
    (void)req; //unused

    // 1) "snapshot" read of current statestore_s
    int seq = stateStoreRef_.g_ui_seq.load(std::memory_order_acquire); // stimcontroller is only one who touches this
    // cast enum -> int
    int stim_window = static_cast<int>(stateStoreRef_.g_ui_state.load(std::memory_order_acquire));
#if CALIB_MODE
    int block_id = stateStoreRef_.g_block_id.load(std::memory_order_acquire);
    int freq_hz_e = static_cast<int>(stateStoreRef_.g_freq_hz_e.load(std::memory_order_acquire));
    int freq_hz = stateStoreRef_.g_freq_hz.load(std::memory_order_acquire);
#else 
    // Defaults (if CALIB_MODE is off)
    int block_id  = 0;
    int freq_hz   = 0;
    int freq_hz_e = 0;
#endif // CALIB_MODE 

    // Run-mode pair 
    int freq_left_hz   = stateStoreRef_.g_freq_left_hz.load(std::memory_order_acquire);
    int freq_right_hz  = stateStoreRef_.g_freq_right_hz.load(std::memory_order_acquire);
    int freq_left_hz_e = static_cast<int>(stateStoreRef_.g_freq_left_hz_e.load(std::memory_order_acquire));
    int freq_right_hz_e= static_cast<int>(stateStoreRef_.g_freq_right_hz_e.load(std::memory_order_acquire));

    // 2) build json string manually
    std::ostringstream oss;
    oss << "{"
        << "\"seq\":"            << seq             << ","
        << "\"stim_window\":"    << stim_window     << ","
        << "\"block_id\":"       << block_id        << ","
        << "\"freq_hz\":"        << freq_hz         << ","
        << "\"freq_hz_e\":"      << freq_hz_e       << ","
        << "\"freq_left_hz\":"   << freq_left_hz    << ","
        << "\"freq_right_hz\":"  << freq_right_hz   << ","
        << "\"freq_left_hz_e\":" << freq_left_hz_e  << ","
        << "\"freq_right_hz_e\":"<< freq_right_hz_e
        << "}";

    std::string json_snapshot = oss.str();

    LOG_ALWAYS("HTTP /state snapshot: " << json_snapshot);
    
    // 3) send json back to client through res
    write_json(res, json_snapshot);
}

// handle ui state transition events written by JS
void HttpServer_C::handle_post_event(const httplib::Request& req, httplib::Response& res){
    // Basic content-type check
    auto it = req.headers.find("Content-Type");
    if (it == req.headers.end() || it->second.find("application/json") == std::string::npos) {
        set_cors_headers(res);
        res.status = 415;
        res.set_content("{\"error\":\"content_type\"}", "application/json");
        return;
    }
    
    const std::string& body = req.body;
    UIStateEvent_E ev = UIStateEvent_None;

    auto p = body.find("\"action\"");
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




