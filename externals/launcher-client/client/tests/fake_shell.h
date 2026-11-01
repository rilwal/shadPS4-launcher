// The shell's side of the overlay service (Linux), for the overlay tests: answers LOVg with a
// header and three buffers, and records LOVc.
#ifndef LAUNCHER_CLIENT_TESTS_FAKE_SHELL_H
#define LAUNCHER_CLIENT_TESTS_FAKE_SHELL_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

inline constexpr uint32_t MAGIC = 0x4C4F5632;
inline constexpr uint32_t NEWER = 1u << 31;

inline int make_memfd(size_t size, bool sealed = false) {
	const int fd = ::memfd_create("lc-test", MFD_CLOEXEC | (sealed ? MFD_ALLOW_SEALING : 0u));
	(void)!::ftruncate(fd, static_cast<off_t>(size));
	if (sealed) {
		// As the shell's: fixed at its size
		(void)!::fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL);
	}
	return fd;
}

class FakeShell {
public:
	// Serves `buffers` (dma-bufs, or memfds standing in for them) of `stride` bytes a row; with
	// none, three memfds of 64x64
	// `sealed_header` false makes a header the game should refuse
	explicit FakeShell(std::vector<int> buffers = {}, uint32_t stride = 256, bool sealed_header = true): m_stride(stride) {
		m_name   = "lc-test.overlay." + std::to_string(::getpid()) + "." + std::to_string(s_count++);
		m_listen = ::socket(AF_UNIX, SOCK_SEQPACKET, 0);
		sockaddr_un addr {};
		addr.sun_family = AF_UNIX;
		std::memcpy(addr.sun_path + 1, m_name.data(), m_name.size());
		::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + m_name.size()));
		::listen(m_listen, 4);

		m_header_fd = make_memfd(16, sealed_header);
		m_header    = static_cast<std::atomic<uint32_t>*>(::mmap(nullptr, 16, PROT_READ | PROT_WRITE, MAP_SHARED, m_header_fd, 0));
		m_header[0] = MAGIC;
		m_header[1] = 1;
		if (buffers.empty()) {
			for (int i = 0; i < 3; i++) {
				buffers.push_back(make_memfd(64 * 64 * 4));
			}
		}
		for (size_t i = 0; i < 3 && i < buffers.size(); i++) {
			m_buffers[i] = buffers[i];
		}
		m_thread = std::thread([this]() { run(); });
	}

	~FakeShell() {
		::shutdown(m_listen, SHUT_RDWR);
		if (m_conn >= 0) {
			::shutdown(m_conn, SHUT_RDWR);
		}
		m_thread.join();
		::munmap(m_header, 16);
		::close(m_header_fd);
		for (int fd: m_buffers) {
			if (fd >= 0) {
				::close(fd);
			}
		}
		if (m_conn >= 0) {
			::close(m_conn);
		}
		::close(m_listen);
	}

	const std::string& name() const { return m_name; }

	// The shell publishes `buffer`: swaps it into the exchange word and takes what was there
	uint32_t publish(uint32_t buffer) { return m_header[1].exchange(buffer | NEWER) & ~NEWER; }
	uint32_t word() const { return m_header[1].load(); }

	// The capture's id, once LOVc came
	std::optional<uint32_t> wait_capture() {
		std::unique_lock lock(m_mutex);
		m_cv.wait_for(lock, std::chrono::seconds(2), [this]() { return m_capture.has_value(); });
		return m_capture;
	}
	bool capture_had_fd() {
		std::lock_guard lock(m_mutex);
		return m_capture_fd;
	}

	std::atomic<int> gets {0};

private:
	void answer() {
		// LOVs: count, then {id, stride, offset, modifier} for the header (10) and buffers 11-13
		std::vector<uint8_t> data {'L', 'O', 'V', 's'};
		const auto put = [&](auto v) {
			const auto* p = reinterpret_cast<const uint8_t*>(&v);
			data.insert(data.end(), p, p + sizeof(v));
		};
		put(uint32_t {4});
		for (uint32_t id = 10; id <= 13; id++) {
			put(id);
			put(uint32_t {id == 10 ? 0u : m_stride});
			put(uint64_t {0});
			put(uint64_t {0});
		}
		const int fds[4] = {m_header_fd, m_buffers[0], m_buffers[1], m_buffers[2]};
		iovec     iov {data.data(), data.size()};
		alignas(cmsghdr) char control[CMSG_SPACE(sizeof(fds))] {};
		msghdr                msg {};
		msg.msg_iov        = &iov;
		msg.msg_iovlen     = 1;
		msg.msg_control    = control;
		msg.msg_controllen = sizeof(control);
		cmsghdr* c         = CMSG_FIRSTHDR(&msg);
		c->cmsg_level      = SOL_SOCKET;
		c->cmsg_type       = SCM_RIGHTS;
		c->cmsg_len        = CMSG_LEN(sizeof(fds));
		std::memcpy(CMSG_DATA(c), fds, sizeof(fds));
		(void)!::sendmsg(m_conn, &msg, MSG_NOSIGNAL);
	}

	void run() {
		m_conn = ::accept(m_listen, nullptr, nullptr);
		if (m_conn < 0) {
			return;
		}
		for (;;) {
			uint8_t               data[64] {};
			iovec                 iov {data, sizeof(data)};
			alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int))] {};
			msghdr                msg {};
			msg.msg_iov        = &iov;
			msg.msg_iovlen     = 1;
			msg.msg_control    = control;
			msg.msg_controllen = sizeof(control);
			const ssize_t n    = ::recvmsg(m_conn, &msg, 0);
			if (n <= 0) {
				return;
			}
			if (n == 4 && std::memcmp(data, "LOVg", 4) == 0) {
				gets++;
				answer();
			} else if (n == 36 && std::memcmp(data, "LOVc", 4) == 0) {
				int fd = -1;
				if (cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr && c->cmsg_type == SCM_RIGHTS) {
					std::memcpy(&fd, CMSG_DATA(c), sizeof(int));
				}
				uint32_t id = 0;
				std::memcpy(&id, data + 4, 4);
				{
					std::lock_guard lock(m_mutex);
					m_capture    = id;
					m_capture_fd = fd >= 0;
				}
				if (fd >= 0) {
					::close(fd);
				}
				m_cv.notify_all();
			}
		}
	}

	std::string             m_name;
	int                     m_listen    = -1;
	int                     m_conn      = -1;
	int                     m_header_fd = -1;
	int                     m_buffers[3] {-1, -1, -1};
	uint32_t                m_stride = 0;
	static inline int       s_count  = 0;
	std::atomic<uint32_t>*  m_header    = nullptr;
	std::thread             m_thread;
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::optional<uint32_t> m_capture;
	bool                    m_capture_fd = false;
};

#endif // LAUNCHER_CLIENT_TESTS_FAKE_SHELL_H
