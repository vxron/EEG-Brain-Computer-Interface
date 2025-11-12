// src/selftest/UnicornSelfTest.cpp
#include <vector>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <array>
#include <cstring>
       
#include "../src/acq/UnicornCheck.h"
#include "../src/utils/Logger.hpp"

// Unicorn SDK header (path provided by CMake include dirs)
extern "C" {
  #include "unicorn.h"
}

constexpr size_t SERIAL_LEN = UNICORN_SERIAL_LENGTH_MAX; 

static bool pick_first_device(UNICORN_DEVICE_SERIAL& out_serial, BOOL onlyPaired, uint32_t& out_count) {
  uint32_t count = 0;
  // Get number of available serials
  int ec = UNICORN_GetAvailableDevices(nullptr, &count, onlyPaired);
  UWARN_IF_FAIL(ec);
  LOG_ALWAYS("GetAvailableDevices(" << (onlyPaired ? "paired" : "all") << ") count=" << count);
  if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;

  // The serials are stored from API as c-style character arrays, so we'll make a vector of all serials
  // data format for Unicorn serials: [14 bytes][14 bytes][14 bytes] (1 serial = 14 char array)
  std::vector<char> flat(count*SERIAL_LEN); // Flat buffer of count x char[14]
  // reinterpret to UNICORN_DEVICE_SERIAL * = char (*)[14]
  auto serials = reinterpret_cast<UNICORN_DEVICE_SERIAL*>(flat.data());

  // This AP call requires first arg to be SPECIFICALLY a ptr to 14 char array
  ec = UNICORN_GetAvailableDevices(serials, &count, onlyPaired);
  UWARN_IF_FAIL(ec);
  if (ec != UNICORN_ERROR_SUCCESS || count == 0) return false;
  
  // Display the list
  // Helper for clipping through c-style arrays
  auto clip_serial = [](const char* s, size_t maxLen) -> std::string {
        size_t n = 0;
        while (n < maxLen && s[n] != '\0') ++n;  // stop at null or maxLen
        return std::string(s, n);
  }; // if this doesn't work -> can try adding null ourselves when making flats or somthn
  
  for (uint32_t i = 0; i<count; i++){
    std::string clipped = clip_serial(&flat[i*SERIAL_LEN],SERIAL_LEN); // starting index for 14-char arrays is incremented by 14 with each iter
    LOG_ALWAYS("Device[" << i << "] serial=" << clipped);
  }

  // Simply select first one and assign to out_serial passed by main caller
  std::memcpy(out_serial, flat.data(), SERIAL_LEN); // copy first serial from flat to out_serial (serial_len=14)
  LOG_ALWAYS("Selected serial=" << clip_serial(out_serial, SERIAL_LEN));
  out_count = count;
  return true;
}

static bool reset_and_enable_eeg_channels_only(UNICORN_HANDLE &handle){
    UNICORN_AMPLIFIER_CONFIGURATION cfg{}; // default init
    UCHECK(UNICORN_GetConfiguration(handle,&cfg));
    // reset
    for(int ch=0;ch<UNICORN_TOTAL_CHANNELS_COUNT;ch++){
        cfg.Channels[ch].enabled = 0;
    }
    // set eeg channels
    for(int ch=UNICORN_EEG_CONFIG_INDEX;ch<(UNICORN_EEG_CONFIG_INDEX+UNICORN_EEG_CHANNELS_COUNT);ch++){
        cfg.Channels[ch].enabled = 1;
    }
    UCHECK(UNICORN_SetConfiguration(handle,&cfg));
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

  // 2) Open the device and set up configs for EEG only
  UNICORN_HANDLE handle{};  // value handle; zero-init
  UCHECK(UNICORN_OpenDevice(serial, &handle));
  LOG_ALWAYS("Device opened.");
  reset_and_enable_eeg_channels_only(handle);
  LOG_ALWAYS("Set up EEG.");

  // 3) Query the number of acquired channels (needed for GetData buffer sizing)
  uint32_t numAcqCh = 0;
  UCHECK(UNICORN_GetNumberOfAcquiredChannels(handle, &numAcqCh));
  LOG_ALWAYS("Acquired channels = " << numAcqCh);

  // 4) Start acquisition in measurement mode (FALSE = not dummy test signal)
  UCHECK(UNICORN_StartAcquisition(handle, FALSE));
  LOG_ALWAYS("Acquisition started (measurement mode).");
  
  // Inspect (make sure) we're running EEG channels
  UNICORN_AMPLIFIER_CONFIGURATION cfg{};
  UCHECK(UNICORN_GetConfiguration(handle, &cfg));
  for (uint32_t i = 0; i < UNICORN_TOTAL_CHANNELS_COUNT; ++i) {
    if (cfg.Channels[i].enabled) {
        LOG_ALWAYS("ENABLED: " << cfg.Channels[i].name
                   << " [" << cfg.Channels[i].unit << "] "
                   << "range=[" << cfg.Channels[i].range[0] << "," << cfg.Channels[i].range[1] << "]");
    }
  }

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
