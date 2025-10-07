#include "../utils/RingBuffer.h"
#include "FakeAcquisition.h"
#include "../utils/Types.h"

// run acq method
// Run method (thread entry)
/*
Accept: a stop request token(or equivalent), a reference to your ring buffer of “Chunk”, and a shared atomic “active frequency” that indicates which flicker the user is “looking at”.

Document that this method loops until a stop is requested, produces exactly one fixed - size chunk per loop, pushes it into the ring buffer, and paces itself based on the chunk period.
*/


/*
* Step 2 — Add the tiny runacq helper (2 functions, simple)

Goal: make the per-iteration code one line, so dropping in SlidingWindow later is trivial.

Create src/acq/RunAcq.h/.cpp:

A single function acquire_one_chunk(IAcqProvider_S&, bufferChunk_S&, std::uint64_t& seq) that:

stamps chunk.seq = seq++, chunk.t0 = now_seconds(), chunk.numCh, chunk.numScans

calls provider.getData(NUM_SCANS_CHUNK, chunk.data.data(), (uint32_t)chunk.data.size())

returns bool.

In main, replace your in-loop stamping + getData with calling this helper. Keep your mean/RMS prints.

Acceptance:

Behavior identical, code in main is shorter and clearer.
*/

class RunAcq_C {
public:
	static bool acquire_one_chunk(IAcqProvider_S& provider, bufferChunk_S& chunk, std::uint64_t& seq) {
		chunk.seq = seq++;
		return provider.getData(NUM_SCANS_CHUNK, chunk.data.data(), static_cast<uint32_t>(chunk.data.size()));
	}
}; // RunAcq_C