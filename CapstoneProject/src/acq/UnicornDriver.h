/* 
Overarching goal: 
- convert getdata() API calls into usable eeg_sample_t formats
- callable by producer thread in main operation to fill ring buffer
- Works with fake data acq provider as well via unicorn_acquire_one_chunk
*/

#include "../utils/RingBuffer.hpp"
#include "FakeAcquisition.h"
#include "../utils/Types.h"
#include "UnicornCheck.h"
#include "IAcqProvider.h"

// include unicorn api somehow

class UnicornDriver_C : public IAcqProvider_S {
public:
	bool unicorn_config_and_open_session(); // establishes unicorn session; sets configuration
	bool unicorn_start_acq(); // start acquisition
	bool unicorn_stop_and_close();
	//bool unicorn_read_one_sample(eeg_sample_t& sample); // uses provider's getdata call to transform into sample format
	bool getData(std::size_t numberOfScans, float* dest, std::uint32_t destLen) override; // single chunk from getdata()

	UnicornDriver_C(const UnicornDriver_C&) = delete;
  	UnicornDriver_C& operator=(const UnicornDriver_C&) = delete;
private:
	unicorn_device_t* m_dev = nullptr; // opaque device/session handle from SDK
  	bool m_running = false;
};

bool UnicornDriver_C::unicorn_config_and_open_session(){

}

bool UnicornDriver_C::unicorn_start_acq(){

}


bool UnicornDriver_C::unicorn_stop_and_close(){

}

bool UnicornDriver_C::getData(std::size_t numberOfScans, float* dest, std::uint32_t destLen){
	return <unicornobject>.getData(NUM_SCANS_CHUNK, dest, static_cast<uint32_t>(dest.size()));
}