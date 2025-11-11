// src/selftest/UnicornSelfTest.cpp
#include <vector>
#include <cstdint>
#include <iostream>
#include <sstream>
       
#include "../src/acq/UnicornCheck.h"
#include "../src/utils/Logger.hpp"

// Unicorn SDK header (path provided by CMake include dirs)
extern "C" {
  #include "unicorn.h"
}

static bool pick_first_device(UNICORN_DEVICE_SERIAL& out_serial, BOOL onlyPaired, uint32_t& out_count) {
  uint32_t count = 0;
  int ec = UNICORN_GetAvailableDevices(nullptr, &count, onlyPaired);
  UWARN_IF_FAIL(ec);
  LOG_ALWAYS("GetAvailableDevices(" << (onlyPaired ? "paired" : "all") << ") count=" << out_count);
  if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;

  std::vector<UNICORN_DEVICE_SERIAL> serials(count);
  ec = UNICORN_GetAvailableDevices(serials.data(), &count, onlyPaired);
  UWARN_IF_FAIL(ec);
  if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;

  // Display the list
  for (uint32_t i = 0; i < count; ++i) {
    LOG_ALWAYS("  Device[" << i << "] serial=" << serials[i]);
  }

  // Simply select first one and assign to out_serial passed by main caller
  std::memcpy(out_serial, serials[0], sizeof(UNICORN_DEVICE_SERIAL)); // copy from serials[0] to out_serial
  LOG_ALWAYS("Selected serial=" << out_serial);
  out_count = count;
  return true;
}

int main() {
  logger::tlabel = "Unicorn";

  // 1) Find a device: try paired first, then any
  UNICORN_DEVICE_SERIAL serial{};
  uint32_t found = 0; // start with 0
  if (!pick_first_device(serial, TRUE, found)) {
    LOG_ALWAYS("No paired devices found. Trying unpaired...");
    if (!pick_first_device(serial, FALSE, found)) {
      LOG_ALWAYS("No devices visible to Windows Bluetooth. Pair the headset first.");
      return 1;
    }
  }

  // 2) Open the device
  UNICORN_HANDLE handle{};  // value handle; zero-init
  UCHECK(UNICORN_OpenDevice(serial, &handle));
  LOG_ALWAYS("Device opened.");

  // 3) Query the number of acquired channels (needed for GetData buffer sizing)
  uint32_t numAcqCh = 0;
  UCHECK(UNICORN_GetNumberOfAcquiredChannels(handle, &numAcqCh));
  LOG_ALWAYS("Acquired channels = " << numAcqCh);

  // 4) Start acquisition in measurement mode (FALSE = not dummy test signal)
  UCHECK(UNICORN_StartAcquisition(handle, FALSE));
  LOG_ALWAYS("Acquisition started (measurement mode).");

  // 5) Grab a small chunk of data
/* GETDATA()
  int UNICORN_GetData(UNICORN_HANDLE hDevice, uint32_t numberOfScans, float *destinationBuffer, uint32_t destinationBufferLength)
  Reads a specific number of scans into the specified destination buffer of known length.
  Checks whether the destination buffer is big enough to hold the requested number of scans.
*/

  const uint32_t scans = 20;
  const uint32_t needed = scans * numAcqCh; // each scan is for all channels
  std::vector<float> buf(needed, 0.0f);

  UCHECK(UNICORN_GetData(handle, scans, buf.data(), needed));
  LOG_ALWAYS("GetData OK: scans=" << scans << ", floats=" << needed);

  // Print the first scan (row)
  if (numAcqCh > 0) {
    std::ostringstream row;
    row << "First scan: ";
    for (uint32_t ch = 0; ch < numAcqCh; ++ch) {
      if (ch) row << ", ";
      row << buf[ch]; // interleaved: scan0 at indices [0..numAcqCh-1]
    }
    LOG_ALWAYS(row.str());
  }

  // 6) Stop & close
  UWARN_IF_FAIL(UNICORN_StopAcquisition(handle));
  UWARN_IF_FAIL(UNICORN_CloseDevice(&handle));
  LOG_ALWAYS("Stopped and closed. Self-test passed.");
  return 0;
}
