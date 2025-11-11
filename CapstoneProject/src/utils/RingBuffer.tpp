#include <cassert>
#include <utility> // std::move

template<typename T>
RingBuffer_C<T>::RingBuffer_C(size_t capacity)
: capacity_(capacity), sem_buffer_slots_available(static_cast<std::ptrdiff_t>(capacity)), sem_data_items_available(0) {
    ringBufferArr.resize(capacity_);
}

template<typename T>
bool RingBuffer_C<T>::push (const T& data) { 
    if(isClosed_.load(std::memory_order_relaxed)) {
        return 0;
    }
    // each time we push, we 'take' a semaphore slot (from the sem_full)
    sem_buffer_slots_available.acquire(); // BLOCKING. producer waits here.
    // should i use try_acquire_until instead? when sem_full is 0 it's time to drain.
    // sem full should only be 'acquirable' again once thing has drained 
    
    // if we get unblocked here from close() call, need to return and release
    if(isClosed_.load((std::memory_order_relaxed))) {
        // must give slot back because we didn't really push anything from here
        sem_buffer_slots_available.release();
        return 0;
    }

    // push to tail
    ringBufferArr[tailIdx_] = data;
    // sems guarantee there's space to add when push happens
    count_++;
    
    // always finish push with tail increment & wrap-around if this increment causes tailIdx_ >= capacity
    tailIdx_ ++; 
    if(tailIdx_ >= capacity_) {
        // wrap-around
        tailIdx_ = 0;
    }

    sem_data_items_available.release(); // "we've pushed something that can be emptied"

    return 1;
}

// pops until an item exists; returns false if closed (BLOCKING)
template<typename T>
bool RingBuffer_C<T>::pop(T *dest) {
    if(isClosed_.load(std::memory_order_relaxed)) {
        return 0;
    }
    sem_data_items_available.acquire(); // need to make sure there's something we can pop
    if(isClosed_.load((std::memory_order_relaxed))) {
        return 0;
    }
    // pop from head, then increment (& consider wrap-around)
    *dest = std::move(ringBufferArr[headIdx_]);
    headIdx_++;
    if(headIdx_ >= capacity_) {
        // wraparound
        headIdx_ = 0;
    }
    sem_buffer_slots_available.release(); // added a slot
    count_--; 
    return 1;
}

// drain: pops as many as currently available items (non-blocking -> skips if can't acquire sem)
// returns num of items
template<typename T>
size_t RingBuffer_C<T>::drain(T* dest) {
    if(isClosed_.load(std::memory_order_acquire)){
        return 0;
    }
    // pop everything from head to tail - continue checking try_acquire() on sem until it fails
    size_t i = 0;
    while(sem_data_items_available.try_acquire()){
        if(isClosed_.load(std::memory_order_relaxed)) {
            break; // if closed, break and return how many you've added before close
        }
        *(dest+i) = std::move(ringBufferArr[headIdx_]);
        headIdx_++;
        count_--;
        // handle wrap around
        if(headIdx_>= capacity_){
            headIdx_ = 0;
        }
        i++;
        sem_buffer_slots_available.release(); // we've got another spot we could write to in the meantime w producer thread
    }

    return i;
}

template<typename T>
void RingBuffer_C<T>::close() {
    isClosed_.store(true, std::memory_order_release);
    // free the semaphores incase they were waiting for an acq
    sem_buffer_slots_available.release();
    sem_data_items_available.release();
}