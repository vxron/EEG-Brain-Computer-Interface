#include "UnicornDriver.h"
#include "../../unicorn/include/unicorn.h"
#include <cstring>
#include <vector>

// Constructor
UnicornDriver_C::UnicornDriver_C(){
    handle = {};
    std::memset(serial, 0, sizeof(serial));
}

// Destructor (RAII: Stop & Close if still open)
UnicornDriver_C::~UnicornDriver_C() {
    if (running) {
        UNICORN_StopAcquisition(handle);
        running = false;
    }
    if (handle) {
        UNICORN_CloseDevice(&handle);
        handle = 0; // makes invalid handle
    }
}

bool UnicornDriver_C::pick_first_device(UNICORN_DEVICE_SERIAL& out_serial, BOOL onlyPaired, uint32_t& out_count) {
  logger::tlabel = "Unicorn Driver";
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
  auto clip_serial = [](const char* s, size_t maxLen) -> std::string {
        size_t n = 0;
        while (n < maxLen && s[n] != '\0') ++n;  // stop at null or maxLen
        return std::string(s, n);
  };

  // Simply select first one and assign to out_serial passed by main caller
  std::memcpy(out_serial, flat.data(), SERIAL_LEN); // copy first serial from flat to out_serial (serial_len=14)
  LOG_ALWAYS("Selected serial=" << clip_serial(out_serial, SERIAL_LEN));
  out_count = count;
  return true;
}

bool UnicornDriver_C::set_configuration(UNICORN_HANDLE &handle){
    UNICORN_AMPLIFIER_CONFIGURATION cfg{}; // default init
    UCHECK(UNICORN_GetConfiguration(handle,&cfg));
    // 1) disable all channels
    for(int ch=0;ch<UNICORN_TOTAL_CHANNELS_COUNT;ch++){
        cfg.Channels[ch].enabled = 0;
    }
    
    // 2) get channel labels and set eeg channels only
    channelLabels_.clear();
    channelLabels_.reserve(UNICORN_EEG_CHANNELS_COUNT);
    
    for(int ch=UNICORN_EEG_CONFIG_INDEX;ch<(UNICORN_EEG_CONFIG_INDEX+UNICORN_EEG_CHANNELS_COUNT);ch++){
        cfg.Channels[ch].enabled = 1;
        // store labels in order
        channelLabels_.emplace_back(cfg.Channels[ch].name); // char name[32] -> unicorn exposes c style array for name
    }
    UCHECK(UNICORN_SetConfiguration(handle,&cfg));
    return true;
}

bool UnicornDriver_C::unicorn_init(){
	logger::tlabel = "Unicorn Driver";
	// 1) Find a device: try paired first, then any
	uint32_t found = 0; // start with 0
	if (!pick_first_device(serial, TRUE, found)) {
		LOG_ALWAYS("No paired devices found. Trying unpaired...");
    	if (!pick_first_device(serial, FALSE, found)) {
      		LOG_ALWAYS("No devices visible to Windows Bluetooth. Pair the headset first.");
      		return false;
    	}
  	}
	// 2) Open the device and set up configs for EEG only
	UCHECK(UNICORN_OpenDevice(serial, &handle));
	LOG_ALWAYS("Device opened.");
	set_configuration(handle);
    numChannels_ = (int)channelLabels_.size();
	LOG_ALWAYS("Set up EEG.");

	return true;
}

bool UnicornDriver_C::unicorn_start_acq(bool testMode){
	logger::tlabel = "Unicorn Driver";
	// Start acquisition in measurement mode (FALSE = not dummy test signal)
	UCHECK(UNICORN_StartAcquisition(handle, testMode ? TRUE : FALSE));
    LOG_ALWAYS("Acquisition started (" << (testMode ? "TEST SIGNAL" : "MEASUREMENT") << ").");
    running = true;
	return true;
}

bool UnicornDriver_C::dump_config_and_indices() {
    logger::tlabel = "Unicorn Driver";

    // Helper: safely print fixed-size char arrays (may not be null-terminated)
    auto clip_cstr = [](const char* s, size_t maxLen) -> std::string {
        if (!s) return std::string();
        size_t n = 0;
        while (n < maxLen && s[n] != '\0') ++n;
        return std::string(s, n);
    };

    UNICORN_AMPLIFIER_CONFIGURATION cfg{}; // default init
    UCHECK(UNICORN_GetConfiguration(handle,&cfg));

    // Print enabled channels + their unit/range
    LOG_ALWAYS("=== Enabled channels (from current configuration) ===");
    for (int i = 0; i < UNICORN_TOTAL_CHANNELS_COUNT; ++i) {
        if (cfg.Channels[i].enabled) {
            const std::string name = clip_cstr(cfg.Channels[i].name, sizeof(cfg.Channels[i].name));
            const std::string unit = clip_cstr(cfg.Channels[i].unit, sizeof(cfg.Channels[i].unit));
            LOG_ALWAYS("EN ch[" << i << "] name=\"" << name
                               << "\" unit=\"" << unit
                               << "\" range=[" << cfg.Channels[i].range[0]
                               << "," << cfg.Channels[i].range[1] << "]");
        }
    }

    uint32_t numAcqCh=0;
    UCHECK(UNICORN_GetNumberOfAcquiredChannels(handle, &numAcqCh));
    LOG_ALWAYS("numAcqCh="<<numAcqCh);

    // Verify index mapping
    LOG_ALWAYS("=== Channel indices within an acquired scan ===");
    const char* names[] = {
        "EEG 1","EEG 2","EEG 3","EEG 4","EEG 5","EEG 6","EEG 7","EEG 8",
        "Battery Level","Counter","Validation Indicator",
        "Accelerometer X","Accelerometer Y","Accelerometer Z",
        "Gyroscope X","Gyroscope Y","Gyroscope Z"
    };

    for (const char* n : names) {
        uint32_t idx = 0;
        int ec = UNICORN_GetChannelIndex(handle, n, &idx);
        if (ec == UNICORN_ERROR_SUCCESS) {
            LOG_ALWAYS("index(\"" << n << "\")=" << idx);
        } else {
            // Useful to see what is NOT present in current acquired scan
            LOG_ALWAYS("index(\"" << n << "\")=N/A (ec=" << ec << ")");
        }
    }

    return true;
}

bool UnicornDriver_C::unicorn_stop_and_close(){
	logger::tlabel = "Unicorn Driver";
	UWARN_IF_FAIL(UNICORN_StopAcquisition(handle));
	UWARN_IF_FAIL(UNICORN_CloseDevice(&handle));
    running = false;
	LOG_ALWAYS("Stopped and closed.");
	return true;
}

// use this to directly write into bufferchunk_s
bool UnicornDriver_C::getData(size_t numberOfScans, float* dest){
	uint32_t numAcqCh = 0; // number of enabled channels
  	UCHECK(UNICORN_GetNumberOfAcquiredChannels(handle, &numAcqCh));
	const uint32_t needed = numberOfScans * numAcqCh;
	UCHECK(UNICORN_GetData(handle, numberOfScans, dest, needed));
	return true;
}