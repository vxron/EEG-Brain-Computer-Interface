#include "../src/stimulus/HttpServer.hpp"
#include "../src/stimulus/StateStore.hpp"
#include "../src/utils/Types.h"

int main() {
    StateStore_s stateStore;
    // init dummy values for test
    stateStore.g_ui_seq.store(42);
    stateStore.g_ui_state.store(UIState_None);   // e.g. “Active_Calib”

    HttpServer_C http(stateStore, 7777);
    http.http_start_server();

    // Blocking listen in this simple test
    http.http_listen_for_poll_requests();

    return 0;
}
