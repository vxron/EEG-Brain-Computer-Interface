// Needs three threads:
// 1- Consumer: main processing pipeline (pulls from ring buffer, and pushes into sliding window) 
// *Consumer should also then process SW by passing it to another function call (window_processor)
// 2 - Producer: main data acquisition pipeline (acquires data from raw api calls and pushes to ring buffer)
// 3 - Stimulus: shows the visual stimuli based on protocol with fsm (calib mode) or just constant (run mode)

#include <thread>
#include <chrono>
#include <iostream>
#include "utils/RingBuffer.hpp"
#include "utils/Types.h"
#include "stimulus/StateStore.hpp"
#include "stimulus/HttpServer.hpp"
#include <atomic>
#include "utils/Logger.hpp"
#include <csignal>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <string_view>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include "acq/UnicornDriver.h"
#if CALIB_MODE
#include <fstream>
#endif

// Global "please stop" flag set by Ctrl+C (SIGINT) to shut down cleanly
static std::atomic<bool> g_stop{false};

#if CALIB_MODE
static std::atomic<bool> g_record{false}; // toggled by stimulus changes during protocol (active vs rest)
#endif

// Interrupt signal sent when ctrl+c is pressed
void handle_sigint(int) {
    g_stop.store(true,std::memory_order_relaxed);
}

void producer_thread_fn(RingBuffer_C<bufferChunk_S>& rb){
    using namespace std::chrono_literals;
    logger::tlabel = "producer";
    LOG_ALWAYS("producer start");
#if ACQ_BACKEND_FAKE
    LOG_ALWAYS("PATH=MOCK");
#else
    LOG_ALWAYS("PATH=HARDWARE");
#endif

	UnicornDriver_C UnicornDriver{};
    if (UnicornDriver.unicorn_init() == false || UnicornDriver.unicorn_start_acq() == false){
        LOG_ALWAYS("unicorn_init failed; exiting producer");
        rb.close();
        return;
    };

    size_t tick_count = 0;
    // ctrl+c check
    while(!g_stop.load(std::memory_order_relaxed)){
        bufferChunk_S chunk{};
        
#if I2C_MOCK
        // simulated sample stream 
        static int16_t i = 0;
        accel_burst_sample.x = i;
        accel_burst_sample.y = i;
        accel_burst_sample.z = i; 
        i++;
        tick_count++;
        accel_burst_sample.tick = tick_count;
        // mimick 119 Hz operation to match accelerometer sampling rate
        // 119Hz = 0.008s period -> PER BURST
        std::this_thread::sleep_for(8ms);
#else
        UnicornDriver.getData(NUM_SCANS_CHUNK, chunk.data.data()); // chunk.data.data() gives type float* (addr of first float in std::array obj)
        tick_count++;
        chunk.tick = tick_count;
#if CALIBRATION_MODE
// does this need a mutex guard or is atomic sufficient?
        chunk.active_label = g_record.load(std::memory_order_acquire); // reader = acquire

#endif // CALIBRATION_MODE
#endif // I2C_MOCK
        
        // blocking push function... will always (eventually) run and return true, 
        // unless queue is closed, in which case we want out
        if(!rb.push(chunk)){
            break;
        } 
    }
    // on thread shutdown, close queue to call release n unblock any consumer waiting for acquire
    rb.close();
}

void consumer_thread_fn(RingBuffer_C<bufferChunk_S>& rb){
    using namespace std::chrono_literals;
    logger::tlabel = "consumer";
    LOG_ALWAYS("consumer start");
    size_t tick_count = 0;

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    bufferChunk_S temp; // placeholder

#if CALIBRATION_MODE
    std::ofstream csv("eeg_calib_data.csv");
    // write first 8 EEG channels of first scan per chunk
    csv << "tick";
    for (std::size_t ch=0; ch<NUM_CH_CHUNK; ++ch) csv << ",eeg" << (ch+1);
    csv << ",active\n";
    size_t rows_written = 0;
#endif

	// build first window
	while(window.sliding_window.get_count()<window.winLen){
		// sc
		if(!rb.pop(&temp)){ // internally wait here (pop cmd is blocking)
			break;
		} 
		else { 
			// this will fit fine because winLen is a multiple of number of scans per channel = 32
			// pop sucessful -> push into sliding window
			for(int i = 0; i<NUM_SAMPLES_CHUNK;i++){
				window.sliding_window.push(temp.data[i]);
			}
		}
	}

    while(!g_stop.load(std::memory_order_relaxed)){
        // (1) emit window to feature extractor
        
        float discard; // first pop
        for(size_t k=0;k<window.winHop;k++){
            window.sliding_window.pop(&discard); 
        }

        while(window.sliding_window.get_count()<window.winLen){ // now push
			std::size_t amnt_left_to_add = window.winLen - window.sliding_window.get_count(); // in samples
			// if there is previous 'len' in stash, we should take it and decrement len
            if (window.stash_len > 0) {
                // take full amnt_left_to_add from stash if it's available, otherwise take window.stash_len
                const std::size_t take = (window.stash_len > amnt_left_to_add) ? amnt_left_to_add : window.stash_len;
                for (std::size_t i = 0; i < take; ++i){
                    window.sliding_window.push(window.stash[i]);
				}
                // move leftover stash to front of array for next round
                if (take < window.stash_len) {
                    std::memmove(window.stash.data(), // start of stash array (dest)
                                 window.stash.data() + take, // ptr to where remaining data starts (src)
                                 (window.stash_len - take) * sizeof(float)); // number of bytes to move
                }
                window.stash_len -= take; // how much we lost for next time; will never be <0

                continue; // go check while-condition again
            }
            // stash is empty
			if(!rb.pop(&temp)){
				break;
			} else {
				// pop successful -> push into sliding window
				if(amnt_left_to_add >= NUM_SAMPLES_CHUNK){
                    for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
                        window.sliding_window.push(temp.data[j]);
                    } // goes back to check while
				}
				else {
					// take what we need and stash the rest for next window
					for(std::size_t j=0;j<NUM_SAMPLES_CHUNK;j++){
						if(j<amnt_left_to_add){
							window.sliding_window.push(temp.data[j]);
						} else { // we know stash should be empty if we made it here
                            if(window.stash_len != 0){
                                LOG_ALWAYS("There's an issue with sliding window stash.");
                                break;
                            }
							window.stash[j-amnt_left_to_add]=temp.data[j];
                            window.stash_len++; // increasing slots to add from stash for next time
						}
					}
				}
			}
            ++tick_count;
            window.tick=tick_count;
		}

#if CALIBRATION_MODE
                if((tick_count%120)==0){
                    LOG_ALWAYS(std::to_string(temp.tick) + " " + std::to_string(temp.x) + " " + std::to_string(temp.y) + " " + std::to_string(temp.z) + " " + std::to_string(temp.active_label) + "\n");
                }
                
                csv << temp.tick << ',' << temp.x << ',' << temp.y << ',' << temp.z << ','
                    << (temp.active_label ? 1 : 0) << '\n';
                rows_written++;
                if(rows_written % 500 == 0) { csv.flush(); } // flush every 500 rows for speed 
#endif
    
	}
    // exiting due to producer exiting means we need to close window rb
    window.sliding_window.close();
    rb.close();
#if CALIBRATION_MODE
    csv.flush();
    csv.close();
#endif
}

void stimulus_thread_fn(){
	// runs protocols in calib mode (handle timing & keep track of state in g_record)
	// should toggle g_record on stimulus switch
	// in calib mode producer should wait for g_record to toggle? or it can just always check state for data ya thats better 0s and 1s in known fixed order...
    
}

int main() {
    LOG_ALWAYS("start (VERBOSE=" << logger::verbose() << ")");

    // Shared singletons/objects
    RingBuffer_C<bufferChunk_S> ringBuf(ACQ_RING_BUFFER_CAPACITY);
    StateStore_s stateStore;
    HttpServer_C server(stateStore);

    // interrupt caused by SIGINT -> 'handle_singint' acts like ISR (callback handle)
    std::signal(SIGINT, handle_sigint);

    // START THREADS. We pass the ring buffer by reference (std::ref) becauase each thread needs the actual shared 'ringBuf' instance, not just a copy...
    // This builds a new thread that starts executing immediately, running producer_thread_rn in parallel with the main thread (same for cons)
    std::thread prod(producer_thread_fn,std::ref(ringBuf));
    std::thread cons(consumer_thread_fn,std::ref(ringBuf));
    std::thread stim(stimulus_thread_fn);

    // Poll the atomic flag g_stop; keep sleep tiny so Ctrl-C feels instant
    while(g_stop.load(std::memory_order_acquire) == 0){
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }

    // on system shutdown:
    // ctrl+c called...
    ringBuf.close();
    prod.join();
    cons.join(); // close "join" individual threads
#if CALIBRATION_MODE
    stim.join();
#endif
    return 0;
}