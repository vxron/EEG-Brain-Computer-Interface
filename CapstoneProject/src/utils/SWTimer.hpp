#pragma once
#include <cstddef> // for size_t
#include <chrono>

class SW_Timer_C {

public:
	using clock_t = std::chrono::steady_clock;
	using dur_t = clock_t::duration;
	using timepoint_t = clock_t::time_point;

	// default duration if caller omits argument
	static constexpr auto DEFAULT = std::chrono::milliseconds{ 15 };

	void start_timer(dur_t timer_dur = DEFAULT) {
		// timeout should occur in timer_dur time from now (if there is a timeout)
		timer_start_time = clock_t::now();
		until = timer_start_time + timer_dur;
		timer_started = true;
	}

	// stop & return elapsed time in ms
	std::chrono::milliseconds stop_timer() {
		auto ended_at = get_timer_value_ms();
		timer_started = false;
		return ended_at;
	}

	// elapsed ms since start (0ms if not started)
	std::chrono::milliseconds get_timer_value_ms() const {
		if (timer_started == false) {
			return std::chrono::milliseconds{ 0 };
		}
		else {
			return std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - timer_start_time);
		}
	}

	bool check_timer_expired() const {
		if (timer_started == true &&
			clock_t::now() >= until) {
			return true;
		}
		else {
			return false;
		}
	}

	bool is_started() const { return timer_started; }

private:
	bool timer_started = false;
	// default-construct timepoint to represent timeout time
	timepoint_t until{};
	timepoint_t timer_start_time{};
	
};