/*
HTTP SERVER : READER
- starts httplib::Server (talks to html) 127.0.0.1
- spends its life waiting for HTML/JS page requests and answering them by reading shared state that stimuluscontroller (c++) updates
- blocks inside listen() loop, hence requires its own thread to prevent freezes
- CLIENT (HTML/JS) calls GET/POST: HTTP server listens & responds...
--> Client calls GET /state on a timer to pull the latest phase/side/freq etc from C++
--> Client calls POST /event when the user clicks a button (e.g. pause/skip or when the page wants to send any other info like "i'm ready" or measured refresh rate)
--> Client calls POST /ready once on load
- server handles all these requests by returning JSON for GET, flipping control flags for POST based on event read 
*/
#pragma once
#include "../../cpp-httplib/httplib.h"
#include "../utils/Logger.hpp"
#include "StateStore.hpp"

// LEARN HOW TO SSH INTO SERVER FROM VS CODE
// then install apt install nginx (preferred web server)
// should see var/www/html# --> root of site you're serving
// https://www.youtube.com/watch?v=RzdM54i7buY

/*
Agreed upon JSON schema for GET /state:

{
  "seq": int,              // monotonic counter, increments when StimulusController publishes new state
  "stim_window": int,      // cast of StimulusState_E (instructions, active, none, run_mode)
  "block_id": int,         // current training block index (0 if none or not calibratio)
  "freq_hz": int,          // stimulus frequency in Hz (e.g., 12)
  "freq_code": int         // cast of TestFreq_E (e.g., 0=7.5Hz,1=10Hz,...)
}
*/


class HttpServer_C {
public: // API
    explicit HttpServer_C(StateStore_s& stateStoreRef, int port=7777);
    ~HttpServer_C();
    bool http_start_server(); // constructs httplib::server
    bool http_listen_for_poll_requests(); // blocking .listen()
    bool http_close_server(); // calls server's stop hook so .listen() returns
    bool get_is_running() { return is_running_; };
private:
    StateStore_s& stateStoreRef_; // reference to StateStore
    httplib::Server* liveServerRef_; // live httplib::server
    int port_;
    std::atomic<bool> is_running_ = false;
    // Handlers
    void handle_get_state(const httplib::Request& req, httplib::Response& res); // return current c++ statestore snapshot (JSON)
    void handle_post_event(const httplib::Request& req, httplib::Response& res); // accept small UI cmds (JSON)
    void handle_post_ready(const httplib::Request& req, httplib::Response& res); // accept page telemetry (refresh Hz)
    void handle_options_and_set(const httplib::Request& req, httplib::Response& res); // CORS preflight
    void write_json(httplib::Response& res, std::string_view json_body) const;
}; // HttpServer_C