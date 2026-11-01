// macOS: the shell's bootstrap service hands out IOSurfaces as Mach ports, so none of them is a
// global IOSurface any process could look up. The messages are Mach messages whose ids are four
// ASCII bytes (docs/plugin-api.md, "Composited overlay").
#include "transport.h"

#if defined(__APPLE__)

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <mach/mach.h>
#include <servers/bootstrap.h>


lc_surface::~lc_surface() {
	if (iosurface != nullptr) {
		CFRelease(static_cast<IOSurfaceRef>(iosurface));
	}
}

namespace lc::transport {

static constexpr mach_msg_id_t MSG_GET      = 0x4C4F5667; // 'LOVg': send the surfaces to the reply port
static constexpr mach_msg_id_t MSG_SURFACES = 0x4C4F5673; // 'LOVs': the header, then the buffers; none if refused
static constexpr mach_msg_id_t MSG_CAPTURE  = 0x4C4F5663; // 'LOVc': the game's capture

static constexpr mach_msg_timeout_t TIMEOUT_MS        = 500;
static constexpr OSType             PIXEL_FORMAT_BGRA = 0x42475241; // 'BGRA'
static constexpr size_t             MAX_PORTS         = 4;

namespace {

struct PortsMsg {
	mach_msg_header_t          header;
	mach_msg_body_t            body;
	mach_msg_port_descriptor_t ports[MAX_PORTS];
};

struct ReceivedMsg {
	PortsMsg               msg;
	mach_msg_max_trailer_t trailer;
};

} // namespace

bool supported() {
	return true;
}

bool Service::connect(const std::string& name, std::string& error) {
	if (name == m_name && m_port != MACH_PORT_NULL) {
		return true;
	}
	disconnect();
	if (name.empty()) {
		return true;
	}
	mach_port_t port = MACH_PORT_NULL;
	if (const kern_return_t kr = bootstrap_look_up(bootstrap_port, name.c_str(), &port); kr != KERN_SUCCESS) {
		error = "couldn't look up " + name + ": " + bootstrap_strerror(kr);
		return false;
	}
	m_port = port;
	m_name = name;
	return true;
}

void Service::disconnect() {
	if (m_port != MACH_PORT_NULL) {
		mach_port_deallocate(mach_task_self(), m_port);
		m_port = MACH_PORT_NULL;
	}
	m_name.clear();
}

bool Service::connected() const {
	return m_port != MACH_PORT_NULL;
}

std::optional<Surfaces> Service::fetch(const std::vector<uint32_t>& buffer_ids, uint32_t header_id, uint32_t width, uint32_t height,
                                       std::string& error) {
	const size_t count = buffer_ids.size() + (header_id != 0 ? 1 : 0);
	if (m_port == MACH_PORT_NULL || count == 0 || count > MAX_PORTS) {
		error = m_port == MACH_PORT_NULL ? "no service" : "wrong number of surfaces";
		return std::nullopt;
	}

	mach_port_t reply = MACH_PORT_NULL;
	if (mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &reply) != KERN_SUCCESS) {
		error = "couldn't make a reply port";
		return std::nullopt;
	}

	ReceivedMsg msg {};
	msg.msg.header.msgh_bits        = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
	msg.msg.header.msgh_size        = sizeof(mach_msg_header_t);
	msg.msg.header.msgh_remote_port = static_cast<mach_port_t>(m_port);
	msg.msg.header.msgh_local_port  = reply;
	msg.msg.header.msgh_id          = MSG_GET;
	const kern_return_t kr = mach_msg(&msg.msg.header, MACH_SEND_MSG | MACH_RCV_MSG | MACH_SEND_TIMEOUT | MACH_RCV_TIMEOUT,
	                                  sizeof(mach_msg_header_t), sizeof(msg), reply, TIMEOUT_MS, MACH_PORT_NULL);
	// A late answer goes with the port
	mach_port_mod_refs(mach_task_self(), reply, MACH_PORT_RIGHT_RECEIVE, -1);
	if (kr != KERN_SUCCESS) {
		error = std::string("no answer from the launcher: ") + mach_error_string(kr);
		return std::nullopt;
	}

	const mach_msg_header_t& header = msg.msg.header;
	const uint32_t got = (header.msgh_bits & MACH_MSGH_BITS_COMPLEX) != 0 ? msg.msg.body.msgh_descriptor_count : 0;
	std::vector<std::unique_ptr<lc_surface>> surfaces;
	if (header.msgh_id == MSG_SURFACES && got == count) {
		for (size_t i = 0; i < count; i++) {
			const mach_msg_port_descriptor_t& port = msg.msg.ports[i];
			// The header comes first
			const uint32_t want = header_id != 0 ? (i == 0 ? header_id : buffer_ids[i - 1]) : buffer_ids[i];
			IOSurfaceRef   ref  = port.type == MACH_MSG_PORT_DESCRIPTOR ? IOSurfaceLookupFromMachPort(port.name) : nullptr;
			if (ref == nullptr || IOSurfaceGetID(ref) != want) {
				error = "the launcher sent other surfaces than it named";
				if (ref != nullptr) {
					CFRelease(ref);
				}
				break;
			}
			// The buffers (not the header) are what the emulator imports at the size overlay_surface gave
			const bool header = header_id != 0 && i == 0;
			if (!header && (IOSurfaceGetPixelFormat(ref) != PIXEL_FORMAT_BGRA || IOSurfaceGetWidth(ref) != width ||
			                IOSurfaceGetHeight(ref) != height)) {
				error = "IOSurface " + std::to_string(want) + " isn't " + std::to_string(width) + "x" + std::to_string(height) + " BGRA";
				CFRelease(ref);
				break;
			}
			auto surface       = std::make_unique<lc_surface>();
			surface->iosurface = ref;
			surface->id        = want;
			surface->stride    = static_cast<uint32_t>(IOSurfaceGetBytesPerRow(ref));
			surfaces.push_back(std::move(surface));
		}
	} else {
		error = got != 0 ? "the launcher's answer doesn't match" : "the launcher refused the surfaces";
	}
	// The surfaces hold their own references
	mach_msg_destroy(&msg.msg.header);
	if (surfaces.size() != count) {
		return std::nullopt;
	}

	Surfaces out;
	size_t   next = 0;
	if (header_id != 0) {
		out.header = std::move(surfaces[next++]);
	}
	for (; next < surfaces.size(); next++) {
		out.buffers.push_back(std::move(surfaces[next]));
	}
	return out;
}

bool Service::send_capture(const lc_capture& capture, uint32_t /*id*/, std::string& error) {
	if (m_port == MACH_PORT_NULL || capture.iosurface == nullptr) {
		error = m_port == MACH_PORT_NULL ? "no service" : "no capture";
		return false;
	}
	PortsMsg msg {};
	msg.header.msgh_bits           = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) | MACH_MSGH_BITS_COMPLEX;
	msg.header.msgh_size           = sizeof(mach_msg_header_t) + sizeof(mach_msg_body_t) + sizeof(mach_msg_port_descriptor_t);
	msg.header.msgh_remote_port    = static_cast<mach_port_t>(m_port);
	msg.header.msgh_id             = MSG_CAPTURE;
	msg.body.msgh_descriptor_count = 1;
	msg.ports[0].name              = IOSurfaceCreateMachPort(static_cast<IOSurfaceRef>(capture.iosurface));
	msg.ports[0].disposition       = MACH_MSG_TYPE_MOVE_SEND;
	msg.ports[0].type              = MACH_MSG_PORT_DESCRIPTOR;
	if (const kern_return_t kr = mach_msg(&msg.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT, msg.header.msgh_size, 0, MACH_PORT_NULL,
	                                      TIMEOUT_MS, MACH_PORT_NULL);
	    kr != KERN_SUCCESS) {
		// Not sent, so the surface's right is still ours
		mach_port_deallocate(mach_task_self(), msg.ports[0].name);
		error = std::string("couldn't send the capture: ") + mach_error_string(kr);
		return false;
	}
	return true;
}

// capture_surface names an IOSurface by its own ID
uint32_t capture_id(const lc_capture& capture) {
	return capture.iosurface != nullptr ? IOSurfaceGetID(static_cast<IOSurfaceRef>(capture.iosurface)) : 0;
}

void adopt_capture(const lc_capture& capture) {
	if (capture.iosurface != nullptr) {
		CFRetain(capture.iosurface);
	}
}

void release_capture(const lc_capture& capture) {
	if (capture.iosurface != nullptr) {
		CFRelease(capture.iosurface);
	}
}

bool HeaderMap::map(const lc_surface& surface) {
	unmap();
	auto* ref = static_cast<IOSurfaceRef>(surface.iosurface);
	if (ref == nullptr || IOSurfaceGetAllocSize(ref) < SIZE) {
		return false;
	}
	CFRetain(ref);
	m_surface = ref;
	return true;
}

void HeaderMap::unmap() {
	if (m_surface != nullptr) {
		CFRelease(m_surface);
		m_surface = nullptr;
	}
}

bool HeaderMap::mapped() const {
	return m_surface != nullptr;
}

// Locked around each access, as the shell's side is
std::atomic<uint32_t>* HeaderMap::lock() {
	auto* ref = static_cast<IOSurfaceRef>(m_surface);
	if (ref == nullptr || IOSurfaceLock(ref, kIOSurfaceLockAvoidSync, nullptr) != kIOReturnSuccess) {
		return nullptr;
	}
	return static_cast<std::atomic<uint32_t>*>(IOSurfaceGetBaseAddress(ref));
}

void HeaderMap::unlock() {
	if (m_surface != nullptr) {
		IOSurfaceUnlock(static_cast<IOSurfaceRef>(m_surface), kIOSurfaceLockAvoidSync, nullptr);
	}
}

} // namespace lc::transport

#endif
