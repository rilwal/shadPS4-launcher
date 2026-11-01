// Platforms without an overlay backend yet (Windows): the shell's window comes to the front instead.
#include "transport.h"

#if !defined(__linux__) && !defined(__APPLE__)

#ifndef _WIN32
#include <unistd.h>
#endif

lc_surface::~lc_surface() {
#ifndef _WIN32
	if (fd >= 0) {
		::close(fd);
	}
#endif
}

namespace lc::transport {

bool supported() {
	return false;
}

bool Service::connect(const std::string& name, std::string& error) {
	error = "not supported on this platform";
	return name.empty();
}

void Service::disconnect() {}

bool Service::connected() const {
	return false;
}

std::optional<Surfaces> Service::fetch(const std::vector<uint32_t>& /*buffer_ids*/, uint32_t /*header_id*/, uint32_t /*width*/,
                                       uint32_t /*height*/, std::string& error) {
	error = "not supported on this platform";
	return std::nullopt;
}

bool Service::send_capture(const lc_capture& /*capture*/, uint32_t /*id*/, std::string& error) {
	error = "not supported on this platform";
	return false;
}

uint32_t capture_id(const lc_capture& /*capture*/) {
	return 1;
}

void adopt_capture(const lc_capture& /*capture*/) {}

void release_capture(const lc_capture& capture) {
#ifndef _WIN32
	if (capture.fd >= 0) {
		::close(capture.fd);
	}
#else
	(void)capture;
#endif
}

bool HeaderMap::map(const lc_surface& /*surface*/) {
	return false;
}

void HeaderMap::unmap() {}

bool HeaderMap::mapped() const {
	return false;
}

std::atomic<uint32_t>* HeaderMap::lock() {
	return nullptr;
}

void HeaderMap::unlock() {}

} // namespace lc::transport

#endif
