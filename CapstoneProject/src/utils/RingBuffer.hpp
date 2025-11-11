/*
==============================================================================
	File: RingBuffer.h
	Desc: Non-Blocking single-producer, single-consumer (SPSC) ring buffer
	for passing data between two threads.
	This header is used by:
  - Acquisition (producer): reads short "Chunks" and stores them in this SPSC
  ring buffer.
  - Decoder (consumer): gets handed chunks from the buffer.

==============================================================================
*/

/*
* FIXED SIZE RING BUFFER WITH SEMAPHORE BASED FULL/EMPTY EVENT SIGNALLING
methods required:
* pop
* push
* drain
* construct(LENGTH)
data required:
* sem_is_empty
* sem_is_full
* capacity
* tail_idx
* head_idx
* data[] -> array (fixed-size at compile time, no heap) of accel samples 
notes:
- works for single producer/single consumer only (no mutual exclusion guarantee for multithread access)
- implemented for compile-time control of capacity (not runtime)
- it is blocking, meaning when full, it waits for pop before push is allowed
*/
#pragma once
#include <semaphore>
#include <cstddef> // std::size_t
#include <vector>
#include <utility>
#include <atomic>

// semaphore template parameter type is ptrdiff_t (type for the update param in the release() function)
// setting its max to 250 means we're saying, 250 is the max we can release at once (really we'll keep it at 1 for this)
static constexpr std::ptrdiff_t SEM_BUFFER_CAPACITY = 250;

// "T" will be eeg sample reads for this application (defined in types.h)
template<typename T>
class RingBuffer_C {
public:

    // a semaphore's count gives current number of allowed allocated 'slots' that can be 'taken'
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_buffer_slots_available;
    std::counting_semaphore<SEM_BUFFER_CAPACITY> sem_data_items_available;

    // Constructor
    explicit RingBuffer_C(size_t capacity);

    // ring buffer methods
    bool pop(T *dest);
    bool push(const T& data);
    size_t drain(T *dest);
    void close();
    size_t get_count() const { return count_.load(std::memory_order_acquire); }; 

private:
    size_t const capacity_;
    size_t tailIdx_ = 0;
    size_t headIdx_ = 0;
    std::atomic<size_t> count_ = 0;
    std::atomic<bool> isClosed_ = 0; // open upon init; atomic because both threads use it
    // full/empty conditions based on semaphore logic 
    bool isFull() const { return 1 ? count_.load(std::memory_order_acquire) == capacity_ : 0; };
    // data array
    std::vector<T> ringBufferArr;
};

#include "RingBuffer.tpp"