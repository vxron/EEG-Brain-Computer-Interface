#include "../src/stimulus/HttpServer.hpp"
#include "../src/stimulus/StateStore.hpp"
#include "../src/utils/Types.h"
#include "../src/stimulus/StimulusController.hpp"
#include "../src/utils/Logger.hpp"
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

/* TEST COMPONENTS:
- Tests the StimulusController along with HTTPServer functionality 
- sets up simple 2 thread architecture that we will use in Main proj
- uses StateStore object referencing
- servers /state to HTML/JS client for visual view transitions
*/

// simple global stop flag for Ctrl+C
static std::atomic<bool> g_stop{false};

static void handle_sigint(int) {
    g_stop.store(true);
}

static void set_state_store_init(StateStore_s & stateStore){
    stateStore.g_ui_seq.store(0, std::memory_order_relaxed);
    //stateStore.g_ui_state.store(UIState_None, std::memory_order_relaxed);
    stateStore.g_block_id.store(0, std::memory_order_relaxed);
    stateStore.g_freq_hz.store(0, std::memory_order_relaxed);
    stateStore.g_freq_hz_e.store(TestFreq_None, std::memory_order_relaxed);
    stateStore.g_refresh_hz.store(0, std::memory_order_relaxed);
}

int main() {
    logger::tlabel = "UIStateMachineSelfTest";
    LOG_ALWAYS("UIStateMachineSelfTest starting…");
    std::signal(SIGINT, handle_sigint);

    // (1) init dummy stateStore values for test
    StateStore_s stateStore {};
    set_state_store_init(stateStore);

    // (2) start http server + stim controller
    HttpServer_C http(stateStore, 7777);
    http.http_start_server();
    StimulusController_C stimController(&stateStore);

    // (3) set up 2 threads
    std::thread httpThread([&]() {
        LOG_ALWAYS("HTTP thread: entering listen loop");
        http.http_listen_for_poll_requests(); // blocks until stop()
        // listens for poll requests from js client to read statestore
        LOG_ALWAYS("HTTP thread: listen loop ended");
    });

    std::thread stimThread([&]() {
        LOG_ALWAYS("Stimulus thread: starting UI state machine");
        //stateStore.g_ui_state.store(UIState_Active_Calib, std::memory_order_relaxed);
        stimController.runUIStateMachine();   // while(1) loop
        // changes ui by writing to statestore upon events 
        LOG_ALWAYS("Stimulus thread: UI state machine exited");
    });

    // (4) main thread: wait for ctrl+c
    while (!g_stop.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }

    LOG_ALWAYS("Shutdown requested, closing HTTP server…");
    http.http_close_server();
    stimController.stopStateMachine();

    if (httpThread.joinable()) httpThread.join();
    if (stimThread.joinable()) stimThread.join();

    LOG_ALWAYS("HttpServerSelfTest exiting.");
    return 0;

}
