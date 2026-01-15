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

		timer_paused = false;
		remaining_after_pause_ = std::chrono::milliseconds{0};
		elapsed_frozen_ = std::chrono::milliseconds{0};

	}

	bool pause_timer(){
		if(!timer_paused && timer_started){
			const auto now = clock_t::now();
			auto rem = until - now;
			if (rem < dur_t::zero()) rem = dur_t::zero();

			// convert to duration for storage
			remaining_after_pause_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(rem);
			// freeze elapsed time
			elapsed_frozen_ = std::chrono::duration_cast<std::chrono::milliseconds>(now - timer_start_time);
			// pause it
			timer_paused = true;
			return true;
		} 
		else {
			return false;
		}
	}

	bool unpause_timer(){
		if(timer_paused && timer_started){
			const auto now = clock_t::now();
			until = remaining_after_pause_ + now;

			// we have to shift the start time up by amount we waited so that get_timer_value_ms stays consistent
			// i.e. it should show how much time we have been in the active sense
			timer_start_time = now - elapsed_frozen_; // so timer is now "recentered" as if it were never frozen and has the new until
			timer_paused = false;
			return true;
		}
		else {
			return false;
		}
	}

	// stop & return elapsed time in ms
	std::chrono::milliseconds stop_timer() {
		auto ended_at = get_timer_value_ms();
		timer_started = false;
		timer_paused = false;
		return ended_at;
	}

	// elapsed ms since start (0ms if not started)
	std::chrono::milliseconds get_timer_value_ms() const {
		if (timer_started == false) {
			return std::chrono::milliseconds{ 0 };
		} else if (timer_paused == true) {
			return elapsed_frozen_;
		}
		else {
			return std::chrono::duration_cast<std::chrono::milliseconds>(clock_t::now() - timer_start_time);
		}
	}

	bool check_timer_expired() const {
		if (timer_paused) {
			return false;
		}
		if (timer_started == true &&
			clock_t::now() >= until) {
			return true;
		}
		else {
			return false;
		}
	}

	bool is_started() const { return timer_started; }
	bool is_paused() const { return timer_paused; }

private:
	bool timer_started = false;
	bool timer_paused = false;
	// default-construct timepoint to represent timeout time
	timepoint_t until{};
	timepoint_t timer_start_time{};

	// for paused state
	std::chrono::milliseconds remaining_after_pause_{0};
    std::chrono::milliseconds elapsed_frozen_{0}; // represents elapsed since start at moment of pause
};