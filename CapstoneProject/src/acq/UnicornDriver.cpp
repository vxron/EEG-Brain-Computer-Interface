#include "UnicornDriver.h"
#include "../../unicorn/include/unicorn.h"

bool UnicornDriver_C::unicorn_config_and_open_session() {
    logger::tlabel = "Unicorn";

    uint32_t count = 0;

    // First pass: ask for count of *paired* devices.
    BOOL onlyPaired = TRUE;
    int ec = UNICORN_GetAvailableDevices(nullptr, &count, onlyPaired);
    UWARN_IF_FAIL(ec);
    LOG_ALWAYS("GetAvailableDevices (paired) count=" << count);

    // If none found, try unpaired scan (useful before pairing in Windows BT UI)
    if (ec == UNICORN_ERROR_SUCCESS && count == 0) {
        onlyPaired = FALSE;
        ec = UNICORN_GetAvailableDevices(nullptr, &count, onlyPaired);
        UWARN_IF_FAIL(ec);
        LOG_ALWAYS("GetAvailableDevices (all) count=" << count);
    }
    if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;

    std::vector<UNICORN_DEVICE_SERIAL> serials(count);
    ec = UNICORN_GetAvailableDevices(serials.data(), &count, onlyPaired);
    UWARN_IF_FAIL(ec);
    if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;

    // OPEN FIRST ONE
    //UNICORN_HANDLE handle = nullptr;
    //UCHECK(UNICORN_OpenDevice(serials[0], &handle));
    // ... store handle, query channels/rate, etc.

    return true;
}
