/*
==============================================================================
	File: IAcqProvider.h
	Desc: Abstract class interface for acquisition providers with different
	getData() implementations (i.e., real Unicorn EEG device & fake acquisition 
	provider for testing). getData() is a pure virtual function that must be 
	implemented by the derived classes.
	Note: provider is selected at build time.
==============================================================================
*/

#pragma once
#include <cstddef>   // std::size_t
#include <cstdint>   // uint32_t

struct IAcqProvider_S {
	// Virtual function interface for acquisition providers
	// **need to make sure the providers are derived from this**
	virtual bool getData(std::size_t const numberOfScans, float* dest, uint32_t destLen) = 0; // pure virtual function = 0
	virtual ~IAcqProvider_S() = default; // virtual destructor for proper cleanup of derived classes
	virtual bool start() = 0; // initialize selected backend
	virtual void stop() = 0;  // shutdown selected backend

	bool provider_read_one_sample(eeg_sample_t& sample); // uses provider's getdata call to transform into sample format
}; // IAcqProvider_S

