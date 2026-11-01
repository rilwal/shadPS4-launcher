// A stand-in for launcherd on a Unix socket, for the session and overlay tests: it accepts one
// connection, answers the hello, and records every message the library sends.
#ifndef LAUNCHER_CLIENT_TESTS_FAKE_LAUNCHER_H
#define LAUNCHER_CLIENT_TESTS_FAKE_LAUNCHER_H

#include "json.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

class FakeLauncher {
public:
	// `after_welcome` goes out right after the welcome, before anything else
	explicit FakeLauncher(std::vector<std::string> after_welcome = {}): m_after_welcome(std::move(after_welcome)) {
		char dir[] = "/tmp/lc-test-XXXXXX";
		m_dir      = ::mkdtemp(dir);
		m_path     = m_dir + "/launcherd.sock";
		m_listen   = ::socket(AF_UNIX, SOCK_STREAM, 0);
		sockaddr_un addr {};
		addr.sun_family = AF_UNIX;
		std::strncpy(addr.sun_path, m_path.c_str(), sizeof(addr.sun_path) - 1);
		::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
		::listen(m_listen, 1);
		::setenv("LAUNCHER_ENDPOINT", m_path.c_str(), 1);
		::setenv("LAUNCHER_SESSION", "test-token", 1);
		m_thread = std::thread([this]() { run(); });
	}

	~FakeLauncher() {
		m_quit = true;
		if (m_fd >= 0) {
			::shutdown(m_fd, SHUT_RDWR);
		}
		::shutdown(m_listen, SHUT_RDWR);
		m_thread.join();
		if (m_fd >= 0) {
			::close(m_fd);
		}
		::close(m_listen);
		::unlink(m_path.c_str());
		::rmdir(m_dir.c_str());
	}

	void send(const std::string& json) {
		std::lock_guard lock(m_write_mutex);
		write_frame(json);
	}

	// Waits for a message of `type` past the first `skip` of them; nothing on timeout
	std::optional<lc::json::Value> wait_for(const std::string& type, size_t skip = 0, int timeout_ms = 2000) {
		std::unique_lock lock(m_mutex);
		std::optional<lc::json::Value> found;
		m_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
			size_t seen = 0;
			for (const auto& msg: m_received) {
				if (msg.text("type") == type && seen++ == skip) {
					found = msg;
					return true;
				}
			}
			return false;
		});
		return found;
	}

	size_t count(const std::string& type) {
		std::lock_guard lock(m_mutex);
		size_t          n = 0;
		for (const auto& msg: m_received) {
			n += msg.text("type") == type ? 1 : 0;
		}
		return n;
	}

	std::vector<std::string> types() {
		std::lock_guard          lock(m_mutex);
		std::vector<std::string> out;
		for (const auto& msg: m_received) {
			out.push_back(msg.text("type"));
		}
		return out;
	}

	std::optional<lc::json::Value> hello() {
		std::lock_guard lock(m_mutex);
		return m_hello;
	}

	// Stops reading after the message it's reading, as a stuck launcherd would
	void stop_reading() { m_stalled = true; }

	// Hangs up, as a launcherd that quits would. Only after the connection is made.
	void disconnect() {
		if (m_fd >= 0) {
			::shutdown(m_fd, SHUT_RDWR);
		}
	}

private:
	bool read_all(void* data, size_t len) {
		auto* p = static_cast<char*>(data);
		while (len != 0) {
			const ssize_t n = ::recv(m_fd, p, len, 0);
			if (n <= 0) {
				return false;
			}
			p += n;
			len -= static_cast<size_t>(n);
		}
		return true;
	}

	void write_frame(const std::string& json) {
		const auto  size = static_cast<uint32_t>(json.size());
		std::string frame(reinterpret_cast<const char*>(&size), 4);
		frame += json;
		#ifdef MSG_NOSIGNAL
		(void)!::send(m_fd, frame.data(), frame.size(), MSG_NOSIGNAL);
#else
		(void)!::send(m_fd, frame.data(), frame.size(), 0);
#endif
	}

	void run() {
		// Waits in poll rather than accept: shutting a listening socket down wakes accept on Linux,
		// but not on macOS, where a test that never connects would hang the destructor
		for (;;) {
			if (m_quit) {
				return;
			}
			pollfd p {m_listen, POLLIN, 0};
			if (::poll(&p, 1, 20) > 0) {
				break;
			}
		}
		m_fd = ::accept(m_listen, nullptr, nullptr);
		if (m_fd < 0) {
			return;
		}
		for (bool first = true;; first = false) {
			while (m_stalled && !m_quit) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}
			uint32_t size = 0;
			if (!read_all(&size, 4)) {
				return;
			}
			std::string payload(size, '\0');
			if (!read_all(payload.data(), size)) {
				return;
			}
			auto msg = lc::json::parse(payload);
			if (first) {
				{
					std::lock_guard lock(m_mutex);
					m_hello = msg;
				}
				std::lock_guard lock(m_write_mutex);
				write_frame(R"({"encoding":"json","server":"fake launcherd"})");
				for (const auto& m: m_after_welcome) {
					write_frame(m);
				}
				continue;
			}
			if (msg) {
				std::lock_guard lock(m_mutex);
				m_received.push_back(std::move(*msg));
			}
			m_cv.notify_all();
		}
	}

	std::vector<std::string>       m_after_welcome;
	std::string                    m_dir;
	std::string                    m_path;
	int                            m_listen = -1;
	int                            m_fd     = -1;
	std::thread                    m_thread;
	std::mutex                     m_write_mutex;
	std::mutex                     m_mutex;
	std::condition_variable        m_cv;
	std::vector<lc::json::Value>   m_received;
	std::optional<lc::json::Value> m_hello;
	std::atomic<bool>              m_stalled {false};
	std::atomic<bool>              m_quit {false};
};

#endif // LAUNCHER_CLIENT_TESTS_FAKE_LAUNCHER_H
