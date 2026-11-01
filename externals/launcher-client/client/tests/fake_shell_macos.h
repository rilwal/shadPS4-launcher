// The shell's side of the overlay service (macOS), for the overlay tests: a bootstrap service
// answering LOVg with a header and three buffers as IOSurface Mach ports, and recording LOVc.
#ifndef LAUNCHER_CLIENT_TESTS_FAKE_SHELL_MACOS_H
#define LAUNCHER_CLIENT_TESTS_FAKE_SHELL_MACOS_H

#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <mach/mach.h>
#include <servers/bootstrap.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

inline constexpr uint32_t MAGIC = 0x4C4F5632;
inline constexpr uint32_t NEWER = 1u << 31;

// A BGRA surface, or the header's 16 bytes as 4x1 of 4
inline IOSurfaceRef make_surface(int width, int height) {
	CFMutableDictionaryRef props = CFDictionaryCreateMutable(nullptr, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	const auto put = [&](CFStringRef key, int value) {
		CFNumberRef n = CFNumberCreate(nullptr, kCFNumberIntType, &value);
		CFDictionarySetValue(props, key, n);
		CFRelease(n);
	};
	put(kIOSurfaceWidth, width);
	put(kIOSurfaceHeight, height);
	put(kIOSurfaceBytesPerElement, 4);
	put(kIOSurfacePixelFormat, 0x42475241 /* 'BGRA' */);
	IOSurfaceRef surface = IOSurfaceCreate(props);
	CFRelease(props);
	return surface;
}

class FakeShell {
public:
	FakeShell() {
		m_name = "lc-test.overlay." + std::to_string(::getpid()) + "." + std::to_string(s_count++);
		if (bootstrap_check_in(bootstrap_port, m_name.c_str(), &m_port) != KERN_SUCCESS) {
			std::fprintf(stderr, "fake shell: couldn't check in %s\n", m_name.c_str());
			m_port = MACH_PORT_NULL;
		}
		m_header = make_surface(4, 1);
		with_header([](std::atomic<uint32_t>* words) {
			words[0] = MAGIC;
			words[1] = 1;
		});
		for (auto& buffer: m_buffers) {
			buffer = make_surface(64, 64);
		}
		m_thread = std::thread([this]() { serve(); });
	}

	~FakeShell() {
		m_stop = true;
		m_thread.join();
		if (m_port != MACH_PORT_NULL) {
			mach_port_mod_refs(mach_task_self(), m_port, MACH_PORT_RIGHT_RECEIVE, -1);
		}
		CFRelease(m_header);
		for (auto* buffer: m_buffers) {
			CFRelease(buffer);
		}
	}

	FakeShell(const FakeShell&)            = delete;
	FakeShell& operator=(const FakeShell&) = delete;

	const std::string& name() const { return m_name; }
	uint32_t           header_id() const { return IOSurfaceGetID(m_header); }
	uint32_t           buffer_id(size_t i) const { return IOSurfaceGetID(m_buffers[i]); }
	IOSurfaceRef       buffer(size_t i) const { return m_buffers[i]; }

	// overlay_surface for these surfaces, saying they're `width` wide
	std::string offer(int width = 64) const {
		return R"({"type":"overlay_surface","buffers":[)" + std::to_string(buffer_id(0)) + "," + std::to_string(buffer_id(1)) + "," +
		       std::to_string(buffer_id(2)) + R"(],"header":)" + std::to_string(header_id()) + R"(,"service":")" + m_name +
		       R"(","width":)" + std::to_string(width) + R"(,"height":64})";
	}

	// The shell publishes `buffer`: swaps it into the exchange word and takes what was there
	uint32_t publish(uint32_t buffer) {
		uint32_t old = 0;
		with_header([&](std::atomic<uint32_t>* words) { old = words[1].exchange(buffer | NEWER) & ~NEWER; });
		return old;
	}
	uint32_t word() {
		uint32_t value = 0;
		with_header([&](std::atomic<uint32_t>* words) { value = words[1].load(); });
		return value;
	}

	// The capture's IOSurface ID, once LOVc came
	std::optional<uint32_t> wait_capture() {
		std::unique_lock lock(m_mutex);
		m_cv.wait_for(lock, std::chrono::seconds(2), [this]() { return m_capture.has_value(); });
		return m_capture;
	}

	std::atomic<int> gets {0};

private:
	static constexpr mach_msg_id_t MSG_GET      = 0x4C4F5667;
	static constexpr mach_msg_id_t MSG_SURFACES = 0x4C4F5673;
	static constexpr mach_msg_id_t MSG_CAPTURE  = 0x4C4F5663;

	struct PortsMsg {
		mach_msg_header_t          header;
		mach_msg_body_t            body;
		mach_msg_port_descriptor_t ports[4];
	};
	struct Received {
		PortsMsg               msg;
		mach_msg_max_trailer_t trailer;
	};

	template <typename F>
	void with_header(F f) {
		if (IOSurfaceLock(m_header, 0, nullptr) == kIOReturnSuccess) {
			f(static_cast<std::atomic<uint32_t>*>(IOSurfaceGetBaseAddress(m_header)));
			IOSurfaceUnlock(m_header, 0, nullptr);
		}
	}

	void serve() {
		while (!m_stop && m_port != MACH_PORT_NULL) {
			Received r {};
			const kern_return_t kr = mach_msg(&r.msg.header, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(r), m_port, 100, MACH_PORT_NULL);
			if (kr != KERN_SUCCESS) {
				continue;
			}
			if (r.msg.header.msgh_id == MSG_GET) {
				// The header first, then the buffers, to the reply port
				PortsMsg answer {};
				answer.header.msgh_bits        = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0) | MACH_MSGH_BITS_COMPLEX;
				answer.header.msgh_size        = sizeof(answer);
				answer.header.msgh_remote_port = r.msg.header.msgh_remote_port;
				answer.header.msgh_id          = MSG_SURFACES;
				answer.body.msgh_descriptor_count = 4;
				IOSurfaceRef offer[4] = {m_header, m_buffers[0], m_buffers[1], m_buffers[2]};
				// Counted before the answer goes, so a client that has it sees the count
				gets++;
				for (int i = 0; i < 4; i++) {
					answer.ports[i].name        = IOSurfaceCreateMachPort(offer[i]);
					answer.ports[i].disposition = MACH_MSG_TYPE_MOVE_SEND;
					answer.ports[i].type        = MACH_MSG_PORT_DESCRIPTOR;
				}
				if (mach_msg(&answer.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT, sizeof(answer), 0, MACH_PORT_NULL, 500, MACH_PORT_NULL) != KERN_SUCCESS) {
					mach_msg_destroy(&answer.header);
				}
				continue;
			}
			if (r.msg.header.msgh_id == MSG_CAPTURE && (r.msg.header.msgh_bits & MACH_MSGH_BITS_COMPLEX) != 0 &&
			    r.msg.body.msgh_descriptor_count == 1) {
				if (IOSurfaceRef capture = IOSurfaceLookupFromMachPort(r.msg.ports[0].name)) {
					std::lock_guard lock(m_mutex);
					m_capture = IOSurfaceGetID(capture);
					CFRelease(capture);
					m_cv.notify_all();
				}
			}
			mach_msg_destroy(&r.msg.header);
		}
	}

	inline static int s_count = 0;

	std::string             m_name;
	mach_port_t             m_port = MACH_PORT_NULL;
	IOSurfaceRef            m_header = nullptr;
	IOSurfaceRef            m_buffers[3] {};
	std::atomic<bool>       m_stop {false};
	std::thread             m_thread;
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::optional<uint32_t> m_capture;
};

#endif // LAUNCHER_CLIENT_TESTS_FAKE_SHELL_MACOS_H
