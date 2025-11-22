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
    stateStore.g_ui_state.store(UIState_None, std::memory_order_relaxed);
    stateStore.g_ui_seq.store(0, std::memory_order_relaxed);
    stateStore.g_block_id.store(0, std::memory_order_relaxed);
    stateStore.g_freq_hz.store(0, std::memory_order_relaxed);
    stateStore.g_freq_hz_e.store(TestFreq_None, std::memory_order_relaxed);
    stateStore.g_refresh_hz.store(0, std::memory_order_relaxed); // init (will get read back thru post)
}

static void set_state_store_calib_at_10_hz(StateStore_s & stateStore){
    stateStore.g_ui_state.store(UIState_Active_Calib, std::memory_order_relaxed); // proxy for user pushes button
    stateStore.g_freq_hz.store(10, std::memory_order_relaxed);
    stateStore.g_freq_hz_e.store(TestFreq_10_Hz, std::memory_order_relaxed);
}

static void set_state_store_run_at_10_hz_and_12_hz(StateStore_s & stateStore){
    stateStore.g_ui_state.store(UIState_Active_Run, std::memory_order_relaxed);
    stateStore.g_freq_hz.store(0, std::memory_order_relaxed);
    stateStore.g_freq_hz_e.store(TestFreq_None, std::memory_order_relaxed);
    stateStore.g_freq_left_hz.store(10, std::memory_order_release);
    stateStore.g_freq_left_hz_e.store(TestFreq_10_Hz, std::memory_order_release);
    stateStore.g_freq_right_hz.store(12, std::memory_order_release);
    stateStore.g_freq_right_hz_e.store(TestFreq_12_Hz, std::memory_order_release);
}

static void set_state_store_instructions_for_10_hz(StateStore_s & stateStore){
    stateStore.g_ui_state.store(UIState_Instructions, std::memory_order_relaxed);
    stateStore.g_freq_hz.store(10, std::memory_order_relaxed);
    stateStore.g_freq_hz_e.store(TestFreq_10_Hz, std::memory_order_relaxed);
}

static void bump_seq(StateStore_s& stateStore) {
    int old = stateStore.g_ui_seq.load(std::memory_order_relaxed);
    stateStore.g_ui_seq.store(old + 1, std::memory_order_release);
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

    // (3) set up 3 threads

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

    // simulate what consumer thread would do (altering the state store in response to user post events)
    std::thread testDriver([&]() {
        using namespace std::chrono_literals;

        // (1) get to home w successful conn
        std::this_thread::sleep_for(1000ms);
        stateStore.g_refresh_hz.store(60, std::memory_order_release); // transition none -> home

        // (2) start calib
        std::this_thread::sleep_for(2000ms);
        LOG_ALWAYS("TEST: inject start_calib event");
        stateStore.g_ui_event.store(UIStateEvent_UserPushesStartCalib, std::memory_order_release);
        std::this_thread::sleep_for(30000ms); // 30s to go thru a couple active/calib cycles based on default proto 

        // (3) exit calib
        LOG_ALWAYS("TEST: inject user pushes exit");
        stateStore.g_ui_event.store(UIStateEvent_UserPushesExit, std::memory_order_release);

        // (4) start run
        std::this_thread::sleep_for(4000ms); 
        LOG_ALWAYS("TEST: inject start_run event");
        stateStore.g_ui_event.store(UIStateEvent_UserPushesStartRun, std::memory_order_release);

        std::this_thread::sleep_for(8000ms);
        LOG_ALWAYS("TEST: inject steady run mode"); // todo: freq setting should be controlled from UI controller (fine for now while we don't have infrastructure)
        set_state_store_run_at_10_hz_and_12_hz(stateStore);
        bump_seq(stateStore);
        std::this_thread::sleep_for(20000ms);

        // (5) exit run
        LOG_ALWAYS("TEST: inject user pushes exit");
        stateStore.g_ui_event.store(UIStateEvent_UserPushesExit, std::memory_order_release);
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
    if (testDriver.joinable()) testDriver.join();

    LOG_ALWAYS("HttpServerSelfTest exiting.");
    return 0;

}
