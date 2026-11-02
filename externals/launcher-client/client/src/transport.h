// The shell's overlay service, per platform: where the shared surfaces come from and the capture
// goes back to (docs/plugin-api.md, "Composited overlay"). Linux: an abstract SOCK_SEQPACKET
// socket handing out dma-bufs and the header's memfd as descriptors. macOS: a bootstrap service
// handing out IOSurfaces as Mach ports.
#ifndef LAUNCHER_CLIENT_TRANSPORT_H
#define LAUNCHER_CLIENT_TRANSPORT_H

#include "launcher_overlay.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// A shared surface from the shell (or the header), and its layout: one plane, `stride` bytes a
// row from `offset`, DRM format modifier `modifier`. It owns its descriptor or IOSurface.
struct lc_surface {
	int      fd        = -1;
	void*    iosurface = nullptr;
	uint32_t id        = 0;
	uint32_t stride    = 0;
	uint64_t offset    = 0;
	uint64_t modifier  = 0;

	lc_surface() = default;
	lc_surface(const lc_surface&)            = delete;
	lc_surface& operator=(const lc_surface&) = delete;
	~lc_surface();
};

namespace lc::transport {

// Whether this platform has a backend
bool supported();

struct Surfaces {
	std::vector<std::unique_ptr<lc_surface>> buffers; // In the order of the ids asked for
	std::unique_ptr<lc_surface>              header;  // If a header id was asked for
};

class Service {
public:
	Service() = default;
	~Service() { disconnect(); }
	Service(const Service&)            = delete;
	Service& operator=(const Service&) = delete;

	// Connects to (or looks up) `name`, keeping hold of it until another name comes
	bool connect(const std::string& name, std::string& error);
	void disconnect();
	[[nodiscard]] bool connected() const;

	// Asks for the surfaces, and checks they're the ones with these ids (and, where the platform
	// can tell, that the buffers are `width` x `height` BGRA)
	std::optional<Surfaces> fetch(const std::vector<uint32_t>& buffer_ids, uint32_t header_id, uint32_t width, uint32_t height,
	                              std::string& error);

	// Hands the game's capture over; capture_surface names `id` after this
	bool send_capture(const lc_capture& capture, uint32_t id, std::string& error);

private:
	std::string m_name;
#if defined(__linux__)
	int m_socket = -1;
#elif defined(__APPLE__)
	unsigned int m_port = 0; // A send right to the service (mach_port_t)
#endif
};

// The id capture_surface names a capture by: an IOSurface's own on macOS, else any but 0
uint32_t capture_id(const lc_capture& capture);
// Takes ownership of what the emulator handed over (retains the IOSurface), and lets it go
void adopt_capture(const lc_capture& capture);
void release_capture(const lc_capture& capture);

// The header's words, mapped: [0] 'LOV2', [1] the exchange word, [2] the shell's publish count,
// [3] reserved
class HeaderMap {
public:
	static constexpr uint32_t MAGIC = 0x4C4F5632;
	static constexpr uint32_t NEWER = 1u << 31;
	static constexpr size_t   SIZE  = 16;

	HeaderMap() = default;
	~HeaderMap() { unmap(); }
	HeaderMap(const HeaderMap&)            = delete;
	HeaderMap& operator=(const HeaderMap&) = delete;

	bool map(const lc_surface& surface);
	void unmap();
	[[nodiscard]] bool mapped() const;

	// The words, from lock until unlock; NULL if they can't be had just now
	std::atomic<uint32_t>* lock();
	void                   unlock();

private:
#if defined(__linux__)
	std::atomic<uint32_t>* m_words = nullptr;
#elif defined(__APPLE__)
	void* m_surface = nullptr; // The header's IOSurfaceRef, retained
#endif
};

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));

} // namespace lc::transport

#endif // LAUNCHER_CLIENT_TRANSPORT_H
