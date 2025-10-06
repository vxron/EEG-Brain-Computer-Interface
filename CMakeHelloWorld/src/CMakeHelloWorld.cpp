// CMakeHelloWorld.cpp : Defines the entry point for the application.
// Goal: show a working console app that uses spdlog,
//       prints a few messages with timestamps, sleeps between them,
//       and then waits for Enter so the window doesn't close immediately.
// STAYING SINGLE THREAD FOR NOW = NO RING BUFFER YET.

#include "CMakeHelloWorld.h"
#include "acq/IAcqProvider.h" // IAcqProvider_S interface
#include "utils/Types.h"   // common types
#include <spdlog/spdlog.h>   // main spdlog API (info/warn/error, set_pattern, set_level)
#include <chrono>            // std::chrono::seconds for sleeping
#include <thread>			 // std::this_thread::sleep_for
#include <iostream>          // std::cin.get() to pause on exit
#include <csignal>		     // CTR-C handler (SIGINT)
#include <atomic>			 // std::atomic_bool for thread-safe flag
#include <cstdint>
#include <cstddef>
#include <numeric>			// std::accumulate
#include <cmath>			// std::ceil
#include <memory>			// std::unique_ptr

#ifdef ACQ_BACKEND_FAKE
#include "acq/FakeAcquisition.h" // class FakeAcquisition_C : IAcqProvider_S
#else
#include "acq/UnicornAcq.h" // class UnicornAcq_C : IAcqProvider_S
#endif

// Consts
constexpr double fs = 250.0; // Hz, Unicorn sampling rate
const double CHUNK_PERIOD_S = static_cast<double>(NUM_SCANS_CHUNK) / fs; // s, assumes 250Hz sampling rate
const long CHUNK_PERIOD_MS = static_cast<long>(std::ceil(CHUNK_PERIOD_S * 1000.0)); // ms, round up to nearest ms

using SteadyClock = std::chrono::steady_clock;

// global "stop flag" toggled by ctrl+c
// static bcuz it only needs to be accessed here (no externs from other files)
// ... and it'll still last program duration
static std::atomic<bool> g_stop(false);

// simple signal handler for ctrl+c
// windows sends SIGINT; we set our flag so main can react
static void on_sigint(int) {
	g_stop.store(true, std::memory_order_relaxed);
}

static inline double now_seconds() {
	return std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();
}

int main() try
{
	// Log message format ("pattern")
	spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
	// Log message level - default to info
	spdlog::set_level(spdlog::level::info);
	spdlog::info("Hello from CMake+spdlog");
	spdlog::info("Press ctrl+c to stop.");
	spdlog::warn("This is a warning");

	// Install the Ctrl+C handler
	std::signal(SIGINT, on_sigint);

	// Construct the acquisition provider (as pointer to base class object, re: polymorphism)
	std::unique_ptr<IAcqProvider_S> AcqProvider;

#ifdef ACQ_BACKEND_FAKE
	spdlog::info("Backend: FAKE");
	// use preconstructed pointer to base class object
	AcqProvider = std::make_unique<FakeAcquisition_C>(FakeAcquisition_C::stimConfigs_S{}); // {} = deafault configs

	//FakeAcquisition_C AcqProvider{ FakeAcquisition_C::stimConfigs_S{} }; 
#else
	spdlog::info("Backend: REAL (Unicorn)");
	//UnicornAcq_C AcqProvider; // TODO
#endif

	// Start provider & nullcheck
	if (!AcqProvider || !AcqProvider->start()) {
		spdlog::critical("Failed to start acquisition provider.");
		return 1;
	}

	// Prepare chunk buffer
	bufferChunk_S chunk{}; // {} uses in-class member initializers (defaults)
	uint64_t seq = 0;
	int consecutiveFailures = 0;
	constexpr int kMaxConsecutiveFailures = 3;

	// Main thread waits until ctrl+c flips g_stop; in the meantime, it reads in one chunk of data
	while (!g_stop.load(std::memory_order_relaxed)) {
		chunk.t0 = now_seconds();
		chunk.seq = seq++;

		const bool ok = AcqProvider->getData(NUM_SCANS_CHUNK, chunk.data.data(), static_cast<uint32_t>(chunk.data.size()));
#ifdef ACQ_BACKEND_FAKE
		std::this_thread::sleep_for(std::chrono::milliseconds(CHUNK_PERIOD_MS)); // mimick acquisition time of one chunk
#endif

		if (!ok) {
			consecutiveFailures++;
			spdlog::warn("getData operation failed ({}/{})", consecutiveFailures, kMaxConsecutiveFailures);
			if (consecutiveFailures >= kMaxConsecutiveFailures) {
				spdlog::error("Too many consecutive failures; stopping.");
				break;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}

		consecutiveFailures = 0; // reset if ok
		
		// compute the mean of chunk.data
		const float mean = std::accumulate(chunk.data.begin(), chunk.data.end(), 0.0f) / chunk.data.size();
		spdlog::info("Mean of acquired chunk data: {:.2f} uV", mean);

	}

	// user requested stop ctrl+c --> ask worker to stop
	AcqProvider->stop();
	spdlog::info("ctrl+c detected, asking worker to stop...");

	// when main() ends, worker's destructor will be called, which joins the thread safely
	// destructor joins = when the object goes out of scope, auto-wait for thread to finish

	// Keep the console window open until Enter is pressed
	spdlog::info("Press Enter to exit console...");
	std::cin.get();

	return 0;
}
catch (const std::exception& e)
{
	spdlog::critical("Unhandled exception: {}", e.what());
	std::cout << "Press Enter to exit...";
	std::cin.get();
	return 1;
}
