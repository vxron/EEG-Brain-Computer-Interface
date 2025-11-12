/* 
Overarching goal: 
- convert getdata() API calls into usable eeg_sample_t formats
- callable by producer thread in main operation to fill ring buffer
- Works with fake data acq provider as well via unicorn_acquire_one_chunk
*/

#pragma once
#include "UnicornCheck.h" // Contains API
#include "IAcqProvider.h"
class UnicornDriver_C : public IAcqProvider_S {
public:
	explicit UnicornDriver_C();
	~UnicornDriver_C();
	bool unicorn_init(); // establishes unicorn session; sets configuration
	bool unicorn_start_acq(); // start acquisition
	bool unicorn_stop_and_close();
	//bool unicorn_read_one_sample(eeg_sample_t& sample); // uses provider's getdata call to transform into sample format
	bool getData(std::size_t numberOfScans, float* dest) override; // single chunk from getdata()

	// one instance only: forbid copying and moving
	UnicornDriver_C(const UnicornDriver_C&)            = delete;
	UnicornDriver_C& operator=(const UnicornDriver_C&) = delete;
	UnicornDriver_C(UnicornDriver_C&&)                 = delete;
	UnicornDriver_C& operator=(UnicornDriver_C&&)      = delete;
private:
	static constexpr size_t SERIAL_LEN = UNICORN_SERIAL_LENGTH_MAX;
  	bool running = false;
	// Opaque Device Handles
	UNICORN_HANDLE handle{};
	UNICORN_DEVICE_SERIAL serial{};
	// Helper functions
	static bool reset_and_enable_eeg_channels_only(UNICORN_HANDLE &handle); // disables all channels, then enables 8 EEG channels
	static bool pick_first_device(UNICORN_DEVICE_SERIAL& out_serial, BOOL onlyPaired, uint32_t& out_count); // finds the first serial available and writes to out_serial
};