// CMakeHelloWorld.cpp : Defines the entry point for the application.
// Goal: show a working console app that uses spdlog,
//       prints a few messages with timestamps, sleeps between them,
//       and then waits for Enter so the window doesn't close immediately.

#include "CMakeHelloWorld.h"
#include <spdlog/spdlog.h>   // main spdlog API (info/warn/error, set_pattern, set_level)
#include <chrono>            // std::chrono::seconds for sleeping
#include <thread>            // std::this_thread::sleep_for
#include <iostream>          // std::cin.get() to pause on exit
#include <csignal>		    // CTR-C handler (SIGINT)
#include <atomic>			// std::atomic_bool for thread-safe flag (to be used with std::thread)
// atomic means op happens as one indivisible step from pov of other threads
#include "acq/IAcqProvider.h" // IAcqProvider_S interface
#include "../utils/Types.h" // common types

#ifdef ACQ_BACKEND_FAKE
#include "acq/FakeAcquisition.h" // class FakeAcquisition_C : IAcqProvider_S
#else
#include "acq/UnicornAcq.h" // class UnicornAcq_C : IAcqProvider_S
#endif


// g_stop.load --> has anyone asked me to stop yet

using namespace std::chrono_literals;

// global "stop flag" toggled by ctrl+c
// static bcuz it only needs to be accessed here (no externs from other files)
// ... and it'll still last program duration
static std::atomic<bool> g_stop(false);

// simple signal handler for ctrl+c
// windows sends SIGINT; we set our flag so main can react
void on_sigint(int) {
	// saying: "im asking u to stop now"
	g_stop.store(true, std::memory_order_relaxed);
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

	// Start a background worker using std::jthread
	//     - std::jthread automatically joins in its destructor
	//     - it passes a stop_token to the thread function which we use to break out of its polling loop
	std::jthread worker([](std::stop_token stopToken) {
		spdlog::info("Worker started");
		int tick = 0;

		// loop until someone calls worker.request_stop() OR main thread calls std::jthread's destructor
		while (!stopToken.stop_requested()) {
			spdlog::info("Worker tick {}", tick++);
			std::this_thread::sleep_for(1s); // simulate doing work every second
		}

		spdlog::info("Worker stopping since stop_requested=true");
	});

	// Main thread waits until ctrl+c flips g_stop, but stays responsive.
	// so we can put lightweight status updates here while background thread is running
	while (!g_stop.load(std::memory_order_relaxed)) {
		//spdlog::info("Main thread waiting...");
		std::this_thread::sleep_for(100ms);
	}

	// user requested stop ctrl+c --> ask worker to stop
	spdlog::info("ctrl+c detected, asking worker to stop...");
	worker.request_stop(); // sets the token seen by worker loop

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
