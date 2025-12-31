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
	virtual bool getData(std::size_t const numberOfScans, float* dest) = 0; // pure virtual function = 0
	virtual ~IAcqProvider_S() = default; // virtual destructor for proper cleanup of derived classes
	virtual bool unicorn_init() = 0; // establishes unicorn session; sets configuration
	virtual bool unicorn_start_acq(bool testMode) = 0; // start acquisition
	virtual bool unicorn_stop_and_close() = 0;
	virtual bool dump_config_and_indices() = 0;
	virtual void setActiveStimulus(double fStimHz) { }; // default no-op

	// channel metadata
    virtual int  getNumChannels() const = 0;
    virtual void getChannelLabels(std::vector<std::string>& out) const = 0;
}; // IAcqProvider_S
