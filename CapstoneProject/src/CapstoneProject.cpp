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
#include "acq/WindowConfigs.hpp"
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

void producer_thread_fn(RingBuffer_C<eeg_sample_t>& rb){
    using namespace std::chrono_literals;
    logger::tlabel = "producer";
    LOG_ALWAYS("producer start");
#if ACQ_BACKEND_FAKE
    LOG_ALWAYS("PATH=MOCK");
#else
    LOG_ALWAYS("PATH=HARDWARE");
#endif

    
    lsm9ds1_driver lsm9ds1("/dev/i2c-1", ADDR_XG);
    if (lsm9ds1.lsm9ds1_init() != 0){
        LOG_ALWAYS("lsm9ds1_init failed; exiting producer");
        rb.close();
        return;
    };

    size_t tick_count = 0;
    // ctrl+c check
    while(!g_stop.load(std::memory_order_relaxed)){
        accel_burst_t accel_burst_sample {};
        
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
        lsm9ds1.lsm9ds1_read_burst(OUT_X_L_XL, &accel_burst_sample); // error handle?
        // can this cause problems, is atomic sufficient or do i need to consider semaphore in addition?
        tick_count++;
        accel_burst_sample.tick = tick_count;
#if CALIBRATION_MODE
// does this need a mutex guard or is atomic sufficient?
        accel_burst_sample.active_label = g_record.load(std::memory_order_acquire); // reader = acquire

#endif // CALIBRATION_MODE
#endif // I2C_MOCK
        
        // blocking push function... will always (eventually) run and return true, 
        // unless queue is closed, in which case we want out
        if(!rb.push(accel_burst_sample)){
            break;
        } 
    }
    // on thread shutdown, close queue to call release n unblock any consumer waiting for acquire
    rb.close();
}

// note ctrl+c exit handled on producer side so only one thread reads g_stop
// when we close the ring buffer, [rb.close() above], pop will return false when consumer tries to pop -> clean exit; otherwise, blocking
void consumer_thread_fn(ringBuffer_C<accel_burst_t>& rb){
    using namespace std::chrono_literals;
    logger::tlabel = "consumer";
    LOG_ALWAYS("consumer start");
    size_t tick_count = 0;

    sliding_window_t window; // should acquire the data for 1 window with that many pops n then increment by hop... 
    accel_burst_t temp; // placeholder for accel burst storage 
    /*
    // wait for first signal that we've reached the WINDOW_SAMPLES length in the buffer
    while(rb.get_count() < window.winLen){
        // don't have enough data
        continue;
    }
    */

#if CALIBRATION_MODE
    std::ofstream csv("accel_calib_data.csv");
    csv << "tick,x,y,z,active\n";
    size_t rows_written = 0;
#endif

    // BUILD FIRST WINDOW - do we need a mutex guard here?  i kinda dont think so because no one else is gonna be popping? and consumer/producer push/pop is handled intrinsically by semaphores in ringbuf class
    for(int i=0;i<window.winLen;i++){
        if(!rb.pop(&temp)){ // internally wait here (pop cmd is blocking)
            break; // throw error
        }
        else {
            // pop successful -> push into sliding window
            window.sliding_window.push(temp);
        }
    }

    while(!g_stop.load(std::memory_order_relaxed)){
        // emit window to feature extractor (DEEEP COPY)
        //featureExtractor.readin(window);
        // pop out half of window for 50% hop
        accel_burst_t discard{};
        for(size_t k=0;k<window.winHop;k++){
            window.sliding_window.pop(&discard); 
        }
        // dont think i need get count cuz pressure is handled intrinsically in ring buff
        /*
        // after the first time, we increment by hop size rather than window size (as long as we have hop size available in array, we can pull a window)
        while(rb.get_count() < window.winHop){
            continue;
        }
        */
        // we have enough to make a window from head by adding the hop amount 
        // keep the tail of the sliding window, overwrite the head (older) 
        for(size_t j=0;j<window.winHop;j++){
            if(!rb.pop(&temp)){
                break; // throw error
            }
            else {
                // pop successful -> push into sliding window
                window.sliding_window.push(temp);
                // each successful pop is something we've acquired from rb
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
        }
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

    ringBuffer_C<bufferChunk_S> ringBuf(RING_BUFFER_CAPACITY);

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