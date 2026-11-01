#include "transport.h"

#if defined(__linux__)

#include <cerrno>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <poll.h>
#include <random>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

lc_surface::~lc_surface() {
	if (fd >= 0) {
		::close(fd);
	}
}

namespace lc::transport {

// Messages begin with four ASCII bytes; the rest is little-endian
static constexpr char MSG_GET[4]      = {'L', 'O', 'V', 'g'}; // Send the surfaces
static constexpr char MSG_SURFACES[4] = {'L', 'O', 'V', 's'}; // u32 count, a layout each, the fds
static constexpr char MSG_CAPTURE[4]  = {'L', 'O', 'V', 'c'}; // The capture's layout and fd

static constexpr int    TIMEOUT_MS  = 500;
static constexpr size_t MAX_FDS     = 4;
static constexpr size_t LAYOUT_SIZE = 24; // {id, stride: u32, offset, modifier: u64}

template <typename T>
static void put(std::vector<uint8_t>& out, T value) {
	const size_t at = out.size();
	out.resize(at + sizeof(T));
	std::memcpy(out.data() + at, &value, sizeof(T));
}

template <typename T>
static T get(const uint8_t* at) {
	T value;
	std::memcpy(&value, at, sizeof(T));
	return value;
}

static bool send_msg(int socket, const std::vector<uint8_t>& data, int fd) {
	iovec                 iov {const_cast<uint8_t*>(data.data()), data.size()};
	alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] {};

	msghdr msg {};
	msg.msg_iov    = &iov;
	msg.msg_iovlen = 1;
	if (fd >= 0) {
		msg.msg_control    = control;
		msg.msg_controllen = sizeof(control);
		cmsghdr* c         = CMSG_FIRSTHDR(&msg);
		c->cmsg_level      = SOL_SOCKET;
		c->cmsg_type       = SCM_RIGHTS;
		c->cmsg_len        = CMSG_LEN(sizeof(int));
		std::memcpy(CMSG_DATA(c), &fd, sizeof(int));
	}
	return ::sendmsg(socket, &msg, MSG_NOSIGNAL) == static_cast<ssize_t>(data.size());
}

bool supported() {
	return true;
}

bool Service::connect(const std::string& name, std::string& error) {
	if (name == m_name && m_socket >= 0) {
		return true;
	}
	disconnect();
	if (name.empty()) {
		return true;
	}

	// An abstract socket: its name follows a NUL
	sockaddr_un addr {};
	addr.sun_family = AF_UNIX;
	if (name.size() + 1 > sizeof(addr.sun_path)) {
		error = "the service's name is too long";
		return false;
	}
	std::memcpy(addr.sun_path + 1, name.data(), name.size());
	const auto len = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());

	m_socket = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	if (m_socket < 0 || ::connect(m_socket, reinterpret_cast<const sockaddr*>(&addr), len) != 0) {
		error = "couldn't connect to " + name + ": " + std::strerror(errno);
		disconnect();
		return false;
	}
	// A shell that stops reading mustn't hold up whoever is sending (the session's reader, close)
	const timeval send_timeout {0, TIMEOUT_MS * 1000};
	::setsockopt(m_socket, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
	m_name = name;
	return true;
}

void Service::disconnect() {
	if (m_socket >= 0) {
		::close(m_socket);
		m_socket = -1;
	}
	m_name.clear();
}

bool Service::connected() const {
	return m_socket >= 0;
}

// A dma-buf doesn't say its size or format: the emulator's import does
std::optional<Surfaces> Service::fetch(const std::vector<uint32_t>& buffer_ids, uint32_t header_id, uint32_t /*width*/,
                                       uint32_t /*height*/, std::string& error) {
	const size_t count = buffer_ids.size() + (header_id != 0 ? 1 : 0);
	if (m_socket < 0 || count == 0 || count > MAX_FDS) {
		error = m_socket < 0 ? "no service" : "wrong number of surfaces";
		return std::nullopt;
	}
	if (!send_msg(m_socket, std::vector<uint8_t>(std::begin(MSG_GET), std::end(MSG_GET)), -1)) {
		error = std::string("couldn't ask the launcher: ") + std::strerror(errno);
		return std::nullopt;
	}

	// Retried on a signal, within the same deadline
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(TIMEOUT_MS);
	for (;;) {
		const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
		pollfd     p {m_socket, POLLIN, 0};
		const int  ready = ::poll(&p, 1, static_cast<int>(left > 0 ? left : 0));
		if (ready == 1) {
			break;
		}
		if (ready < 0 && errno == EINTR) {
			continue;
		}
		error = "no answer from the launcher";
		return std::nullopt;
	}

	uint8_t               data[8 + MAX_FDS * LAYOUT_SIZE] {};
	iovec                 iov {data, sizeof(data)};
	alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * MAX_FDS)] {};

	msghdr msg {};
	msg.msg_iov        = &iov;
	msg.msg_iovlen     = 1;
	msg.msg_control    = control;
	msg.msg_controllen = sizeof(control);

	const ssize_t n = ::recvmsg(m_socket, &msg, MSG_CMSG_CLOEXEC);

	// Whatever came is owned here, and closed unless it's what was asked for
	std::vector<std::unique_ptr<lc_surface>> got;
	for (cmsghdr* c = n >= 0 ? CMSG_FIRSTHDR(&msg) : nullptr; c != nullptr; c = CMSG_NXTHDR(&msg, c)) {
		if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS) {
			continue;
		}
		const size_t fds = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
		for (size_t i = 0; i < fds; i++) {
			auto surface = std::make_unique<lc_surface>();
			std::memcpy(&surface->fd, CMSG_DATA(c) + i * sizeof(int), sizeof(int));
			got.push_back(std::move(surface));
		}
	}

	if (n < 8 || std::memcmp(data, MSG_SURFACES, 4) != 0) {
		error = n <= 0 ? "the launcher hung up" : "the launcher's answer doesn't match";
		return std::nullopt;
	}
	const auto listed = get<uint32_t>(data + 4);
	if (listed == 0 || got.empty()) {
		error = "the launcher refused the surfaces";
		return std::nullopt;
	}
	if (listed != count || got.size() != count || static_cast<size_t>(n) < 8 + count * LAYOUT_SIZE) {
		error = "the launcher's answer doesn't match";
		return std::nullopt;
	}

	for (size_t i = 0; i < count; i++) {
		const uint8_t* at      = data + 8 + i * LAYOUT_SIZE;
		lc_surface&    surface = *got[i];
		surface.id             = get<uint32_t>(at);
		surface.stride         = get<uint32_t>(at + 4);
		surface.offset         = get<uint64_t>(at + 8);
		surface.modifier       = get<uint64_t>(at + 16);
		// The header comes first
		const uint32_t want = header_id != 0 ? (i == 0 ? header_id : buffer_ids[i - 1]) : buffer_ids[i];
		if (surface.id != want) {
			error = "the launcher sent other surfaces than it named";
			return std::nullopt;
		}
	}

	Surfaces out;
	size_t   next = 0;
	if (header_id != 0) {
		out.header = std::move(got[next++]);
	}
	for (; next < got.size(); next++) {
		out.buffers.push_back(std::move(got[next]));
	}
	return out;
}

bool Service::send_capture(const lc_capture& capture, uint32_t id, std::string& error) {
	if (m_socket < 0 || capture.fd < 0) {
		error = m_socket < 0 ? "no service" : "no capture";
		return false;
	}
	std::vector<uint8_t> data(std::begin(MSG_CAPTURE), std::end(MSG_CAPTURE));
	put(data, id);
	put(data, capture.width);
	put(data, capture.height);
	put(data, capture.stride);
	put(data, capture.offset);
	put(data, capture.modifier);
	if (!send_msg(m_socket, data, capture.fd)) {
		error = std::string("couldn't send the capture: ") + std::strerror(errno);
		return false;
	}
	return true;
}

uint32_t capture_id(const lc_capture& /*capture*/) {
	const auto id = static_cast<uint32_t>(std::random_device {}());
	return id != 0 ? id : 1;
}

void adopt_capture(const lc_capture& /*capture*/) {}

void release_capture(const lc_capture& capture) {
	if (capture.fd >= 0) {
		::close(capture.fd);
	}
}

bool HeaderMap::map(const lc_surface& surface) {
	unmap();
	struct stat st {};
	if (surface.fd < 0 || ::fstat(surface.fd, &st) != 0 || static_cast<size_t>(st.st_size) < SIZE) {
		return false;
	}
	// Sealed against shrinking, or the shell truncating it would make the next access a SIGBUS
	const int seals = ::fcntl(surface.fd, F_GET_SEALS);
	if (seals < 0 || (seals & F_SEAL_SHRINK) == 0) {
		return false;
	}
	void* addr = ::mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, surface.fd, 0);
	if (addr == MAP_FAILED) {
		return false;
	}
	m_words = static_cast<std::atomic<uint32_t>*>(addr);
	return true;
}

void HeaderMap::unmap() {
	if (m_words != nullptr) {
		::munmap(m_words, SIZE);
		m_words = nullptr;
	}
}

bool HeaderMap::mapped() const {
	return m_words != nullptr;
}

// Mapped for good: the words are plain shared memory
std::atomic<uint32_t>* HeaderMap::lock() {
	return m_words;
}

void HeaderMap::unlock() {}

} // namespace lc::transport

#endif
