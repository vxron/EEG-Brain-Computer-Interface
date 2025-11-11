/*
==============================================================================
	File: MicroComms.h
	Desc: A collection of bit operations that are useful for communicating 
	with the micro for actuation purposes.
	*Assumes 32-bit words.
==============================================================================
*/

#pragma once
#include <cstdint>
#include <string>
#include "Types.h"

struct MicroComms_S {

	// Operation options: "toggle", "set", "clear", "read"
	bool bit_manipulation(uint32_t& data, const BitOperation_E& operation, const uint32_t& bitPos);
	uint32_t read_bit(const uint32_t& data, const uint32_t& bitPos);
	bool reverse_bits_in_each_byte_32(uint32_t& data);
	uint32_t read_bit_range(const uint32_t& data, const uint32_t& bitPosStart, const uint32_t& bitPosEnd);
	bool write_bit(uint32_t& data, const uint32_t& bitPos, const uint32_t& bitToWrite);
	bool write_bit_range(uint32_t& data, const uint32_t& bitPosStart, const uint32_t bitPosEnd, const uint32_t valueToWrite);

}; // MicroComms_S
