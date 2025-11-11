#include "MicroComms.h"

namespace {
	// simple guards
	inline bool valid_bitpos(std::size_t pos) { return pos < 32; }
	inline bool valid_range(std::size_t start, std::size_t end) {
		return start <= end && end < 32;
	}
}

bool MicroComms_S::bit_manipulation(uint32_t& data, const BitOperation_E& operation, const uint32_t& bitPos) {

	uint32_t bitMask = 0x0;

	switch (operation) {
	case BitOp_Toggle:
		bitMask = 0x1 << bitPos;
		data = data ^ bitMask;
		break;
	case BitOp_Set:
		bitMask = 0x1 << bitPos;
		data = data | bitMask;
		break;
	case BitOp_Clear:
		bitMask = ~(0x1 << bitPos); // always shift with 1 (same as other masks), then flip
		data = data & bitMask;
		break;
	default:
		return false;
	}

	return true;
}

uint32_t MicroComms_S::read_bit(const uint32_t& data, const uint32_t& bitPos) {
	if (!valid_bitpos(bitPos)) {
		return 0;
	}
	
	uint32_t bitMask = (0x1 << bitPos) & data;
	if (bitMask == 0x0) {
		return 0b0;
	}
	else {
		return 0b1;
	}
}

// start..end inclusive, 0=MSB, 31=LSB, start <= end
uint32_t MicroComms_S::read_bit_range(const uint32_t& data, const uint32_t& bitPosStart, const uint32_t& bitPosEnd) {
	if (!valid_range(bitPosStart, bitPosEnd)) {
		return 0u;
	}
	uint32_t range = data << bitPosStart;
	range = range >> (32 - bitPosEnd - 1 + bitPosStart);
	return range;
}

bool MicroComms_S::write_bit_range(uint32_t& data, const uint32_t& bitPosStart, const uint32_t bitPosEnd, const uint32_t valueToWrite) {
	if (!valid_range(bitPosStart, bitPosEnd)) {
		return 0u;
	}
	for (int i = bitPosStart; i <= bitPosEnd; i++) {
		write_bit(data, i, (read_bit(valueToWrite, (i - bitPosStart))));
	}
	return true;
}

bool MicroComms_S::write_bit(uint32_t& data, const uint32_t& bitPos, const uint32_t& bitToWrite) {
	if (!valid_bitpos(bitPos)) {
		return 0;
	}
	if (bitToWrite == 0) {
		bit_manipulation(data, BitOp_Clear, bitPos);
	}
	else if (bitToWrite == 1) {
		bit_manipulation(data, BitOp_Set, bitPos);
	}

	return true;
}

// Purpose: Comm protocols like UART generally feed LSB first... need to swap for reading
// how "data" is organized: (assume little endian)
// b0 b1 b2 b3 b4 b5 b6 b7 [byte 4 --> Least sig]
// b0 b1 b2 b3 b4 b5 b6 b7 [byte 3]
// b0 b1 b2 b3 b4 b5 b6 b7 [byte 2]
// b0 b1 b2 b3 b4 b5 b6 b7 [byte 1 --> Most sig]
// need to swap each byte so MSB (b7) is first
bool MicroComms_S::reverse_bits_in_each_byte_32(uint32_t& data) {

	for (int byteIdx = 0; byteIdx < 4; byteIdx++) {
		uint32_t byte = read_bit_range(data, byteIdx * 8, byteIdx * 8 + 7);

		for (int i = 0; i < 4; i++) {
			// Perform swap
			uint32_t val1 = read_bit(byte, i);
			write_bit(byte, i, read_bit(byte, 7 - i));
			write_bit(byte, 7 - i, val1);
		}

		// write this byte to data
		write_bit_range(data, byteIdx * 8, byteIdx * 8 + 7, byte);

	}
	return true;
}