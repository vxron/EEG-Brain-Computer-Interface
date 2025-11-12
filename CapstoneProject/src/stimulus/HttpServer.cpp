

// this must go in a REQUESTTHREAD -> so we constantly control with set & update UI every frame
// (terminate this thread when app should close etc)
// how do we go between the 2 threads? 
// 1) create a mutex around the variable (string) we're wanting to get or set (e.g. frequency) -> lock, unlock, lock between gets/sets on different threads
// 2) ^but that would require a lot of locking/unlocking every frame... instead DOUBLE BUFFER
// have 2 diff strings:
// std::array<std::string,2> m_weatherInfo
// std::atomic<uint64_t> m_weatherInfoIndex <- index into which string we're working with (atomic to make sure we're not reading/writing at same time)
// std::mutex m_WeatherInfoMutex
// common case: just trying to render -> use m_weatherinfoindex to access m_weatherinfo array
// on update: increment current index, write into other index (so we can be reading )
// when its time to update data, use current index and increment so we write into other index (meaning we can still be reading from the other)
// once write completes, we store new idx so then we read/render the new updated data

#include "HttpServer.hpp"
#include <atomic>
#include <string>
#include <sstream>
#include <iomanip>

// Constructor
HttpServer_C::HttpServer_C(StateStore_s& stateStoreRef, int port=7777) : stateStoreRef_(stateStoreRef), liveServerRef_(nullptr), port_(port) {
}

// Helpers
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
void HttpServer_C::handle_options_and_set(const httplib::Request&, httplib::Response& res) {
    set_cors_headers(res);
    res.status = 200;
}

// Handlers

void HttpServer_C::handle_get_state(const httplib::Request&, httplib::Response&){

}
void HttpServer_C::handle_post_event(const httplib::Request&, httplib::Response&){

}
void HttpServer_C::handle_post_ready(const httplib::Request&, httplib::Response&){

}

// Lifecycle
bool HttpServer_C::http_start_server(){
    if (is_running_.load()) return false;

    liveServerRef_ = new httplib::Server(); // listens for requests from HTML/JS

    // Default headers (we also put CORS per response)
    liveServerRef_->set_default_headers({
        {"Access-Control-Allow-Origin","*"},
        {"Access-Control-Allow-Methods","GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers","Content-Type"}
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

    return true;
}

bool HttpServer_C::http_listen_for_poll_requests(){
    
}

bool HttpServer_C::http_close_server(){
    if (!liveServerRef_) return false;
    liveServerRef_->stop(); // breaks .listen()
    return true;
}




