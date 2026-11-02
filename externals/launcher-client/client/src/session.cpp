#include "internal.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace lc {

static constexpr const char* ENV_ENDPOINT = "LAUNCHER_ENDPOINT";
static constexpr const char* ENV_SESSION  = "LAUNCHER_SESSION";

// Minimum time between two updates of one activity
static constexpr auto ACTIVITY_INTERVAL = std::chrono::milliseconds(100);

// Frames bigger than this are a protocol error, not a message
static constexpr uint32_t MAX_FRAME = 16 * 1024 * 1024;

static std::atomic<lc_session*> g_session {nullptr};

static std::string get_env(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr ? std::string(value) : std::string {};
}

void vlog(const lc_callbacks& callbacks, lc_log_level level, const char* format, va_list args) {
	char line[1024];
	std::vsnprintf(line, sizeof(line), format, args);
	if (callbacks.log != nullptr) {
		callbacks.log(callbacks.user, level, line);
	} else {
		std::fprintf(stderr, "[launcher] %s\n", line);
	}
}

static uint32_t process_id() {
#ifdef _WIN32
	return static_cast<uint32_t>(::GetCurrentProcessId());
#else
	return static_cast<uint32_t>(::getpid());
#endif
}

#ifdef _WIN32
static std::wstring widen(const std::string& s) {
	if (s.empty()) {
		return {};
	}
	const int size = ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring out(static_cast<size_t>(size), L'\0');
	::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), size);
	return out;
}

// One read or write on the pipe. It's opened for overlapped I/O because synchronous I/O on a
// handle is serialised: the reader's pending ReadFile would hold up every write.
static bool pipe_io(void* pipe, bool write, void* data, DWORD len, DWORD& done) {
	// One event per thread, kept until the thread exits. ReadFile and WriteFile reset it as they
	// start.
	struct Event {
		HANDLE handle = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
		~Event() {
			if (handle != nullptr) {
				::CloseHandle(handle);
			}
		}
	};
	static thread_local const Event event;
	if (event.handle == nullptr) {
		return false;
	}
	OVERLAPPED ov {};
	ov.hEvent          = event.handle;
	const BOOL started = write ? ::WriteFile(pipe, data, len, nullptr, &ov) : ::ReadFile(pipe, data, len, nullptr, &ov);
	return (started != 0 || ::GetLastError() == ERROR_IO_PENDING) && ::GetOverlappedResult(pipe, &ov, &done, TRUE) != 0 &&
	       done > 0;
}

// The user a process runs as, in `buf`
static PSID user_of(HANDLE process, std::vector<uint8_t>& buf) {
	HANDLE token {};
	if (::OpenProcessToken(process, TOKEN_QUERY, &token) == 0) {
		return nullptr;
	}
	DWORD size = 0;
	::GetTokenInformation(token, TokenUser, nullptr, 0, &size);
	buf.resize(size);
	const bool ok = size != 0 && ::GetTokenInformation(token, TokenUser, buf.data(), size, &size) != 0;
	::CloseHandle(token);
	return ok ? reinterpret_cast<TOKEN_USER*>(buf.data())->User.Sid : nullptr;
}

// Whether the pipe's server runs as we do. Anyone can make a pipe by that name if the launcher
// hasn't, and the session token isn't theirs to see; the launcher checks its clients the same way.
static bool server_is_us(HANDLE pipe) {
	ULONG pid = 0;
	if (::GetNamedPipeServerProcessId(pipe, &pid) == 0) {
		return false;
	}
	const HANDLE server = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
	if (server == nullptr) {
		return false;
	}
	std::vector<uint8_t> ours;
	std::vector<uint8_t> theirs;
	const PSID           us   = user_of(::GetCurrentProcess(), ours);
	const PSID           them = user_of(server, theirs);
	::CloseHandle(server);
	return us != nullptr && them != nullptr && ::EqualSid(us, them) != 0;
}
#endif

} // namespace lc

using namespace lc;

void lc_session::log(lc_log_level level, const char* format, ...) {
	va_list args;
	va_start(args, format);
	vlog(callbacks, level, format, args);
	va_end(args);
}

lc_session::~lc_session() {
#ifdef _WIN32
	if (m_pipe != nullptr) {
		::CloseHandle(m_pipe);
	}
#else
	if (m_fd >= 0) {
		::close(m_fd);
	}
#endif
}

bool lc_session::open_endpoint(const std::string& endpoint, std::string& error) {
#ifdef _WIN32
	// LAUNCHER_ENDPOINT is the pipe's name, \\.\pipe\launcherd-<user>. SECURITY_IDENTIFICATION
	// keeps the launcher from acting as us, which it has no need to.
	const std::wstring name = widen(endpoint);
	for (int attempt = 0;; attempt++) {
		const HANDLE pipe = ::CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
		                                  FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
		if (pipe != INVALID_HANDLE_VALUE) {
			if (!server_is_us(pipe)) {
				error = "not connecting to " + endpoint + ": its server doesn't run as this user, or that couldn't be checked";
				::CloseHandle(pipe);
				return false;
			}
			m_pipe = pipe;
			return true;
		}
		// Every instance is taken by another client until the launcher makes the next one
		const DWORD code = ::GetLastError();
		if (code != ERROR_PIPE_BUSY || attempt == 2 || ::WaitNamedPipeW(name.c_str(), 1000) == 0) {
			error = "could not connect to " + endpoint + ": error " + std::to_string(code);
			return false;
		}
	}
#else
	sockaddr_un addr {};
	addr.sun_family = AF_UNIX;
	if (endpoint.size() >= sizeof(addr.sun_path)) {
		error = "the endpoint's path is too long: " + endpoint;
		return false;
	}
	std::memcpy(addr.sun_path, endpoint.c_str(), endpoint.size() + 1);

#ifdef SOCK_CLOEXEC
	m_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
	m_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
#endif
	if (m_fd < 0 || ::connect(m_fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
		error = "could not connect to " + endpoint + ": " + std::strerror(errno);
		return false;
	}
#ifdef SO_NOSIGPIPE
	const int one = 1;
	::setsockopt(m_fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
	return true;
#endif
}

bool lc_session::wait_readable(int timeout_ms) {
#ifdef _WIN32
	// A pipe can't be waited on for data without starting a read, so look every 10 ms
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	while (std::chrono::steady_clock::now() < deadline) {
		DWORD available = 0;
		if (::PeekNamedPipe(m_pipe, nullptr, 0, nullptr, &available, nullptr) == 0 || available != 0) {
			// Closed (the read that follows says so), or something to read
			return true;
		}
		::Sleep(10);
	}
	return false;
#else
	pollfd pfd {m_fd, POLLIN, 0};
	for (;;) {
		const int n = ::poll(&pfd, 1, timeout_ms);
		if (n < 0 && errno == EINTR) {
			continue;
		}
		return n > 0;
	}
#endif
}

bool lc_session::connect(const lc_connect_info& info, std::string& error) {
	const std::string endpoint = get_env(ENV_ENDPOINT);
	if (!open_endpoint(endpoint, error)) {
		return false;
	}

	// A runner's overlay capability is its game sessions', which draw the overlay themselves: the runner
	// keeps it whatever this platform supports, and has no overlay of its own
	const bool runner       = (info.capabilities & protocol::LC_CAP_RUNNER) != 0;
	uint32_t   capabilities = info.capabilities;
	if (!runner && !lc_overlay_supported()) {
		capabilities &= ~static_cast<uint32_t>(LC_CAP_COMPOSITED_OVERLAY);
	}
	const std::string name  = info.name != nullptr ? info.name : "";
	const auto        hello = protocol::hello(get_env(ENV_SESSION), LC_PLUGIN_API, capabilities, name);
	if (!write_frame(hello.json)) {
		error = "could not send the hello";
		return false;
	}
	m_connected_at = std::chrono::steady_clock::now();

	// launcherd answers straight away; don't hold up the boot if it doesn't
	const int   timeout = info.timeout_ms != 0 ? static_cast<int>(info.timeout_ms) : 3000;
	std::string frame;
	if (!wait_readable(timeout)) {
		error = "no welcome from the launcher";
		return false;
	}
	if (!read_frame(frame)) {
		error = "the launcher closed the connection during the handshake";
		return false;
	}
	const auto welcome = protocol::parse_host(frame);
	if (welcome.encoding != "json") {
		error = "the launcher chose an encoding the library doesn't speak: " + frame;
		return false;
	}

	if (!runner && (capabilities & LC_CAP_COMPOSITED_OVERLAY) != 0) {
		overlay = overlay_create(*this);
	}
	log(LC_LOG_INFO, "connected to %s at %s", welcome.server.c_str(), endpoint.c_str());
	return true;
}

void lc_session::start() {
	m_reader = std::thread([this]() { reader(); });
	m_writer = std::thread([this]() { writer(); });
}

void lc_session::shutdown_io() {
#ifdef _WIN32
	if (m_pipe != nullptr) {
		::CancelIoEx(m_pipe, nullptr);
	}
#else
	if (m_fd >= 0) {
		::shutdown(m_fd, SHUT_RDWR);
	}
#endif
}

void lc_session::close() {
	if (overlay != nullptr) {
		overlay_stop(overlay);
	}
	const bool flushed = flush(std::chrono::seconds(1));
	{
		std::lock_guard lock(m_queue_mutex);
		m_stop = true;
	}
	m_queue_cv.notify_all();
	// A launcherd that stopped reading leaves the writer blocked in a write, which only taking the
	// connection down ends. That, too, has to be repeated on Windows, for a write not yet started.
	const auto until_done = [this](std::condition_variable& cv, std::mutex& mutex, const bool& done) {
		for (;;) {
			shutdown_io();
			std::unique_lock lock(mutex);
			if (cv.wait_for(lock, std::chrono::milliseconds(20), [&]() { return done; })) {
				return;
			}
		}
	};
	if (m_writer.joinable()) {
		if (!flushed) {
			until_done(m_flushed_cv, m_queue_mutex, m_writer_done);
		}
		m_writer.join();
	}
	if (m_reader.joinable()) {
		// On Windows a read that hadn't started when the I/O was cancelled would still wait, so
		// cancel until the reader has seen it
		until_done(m_configure_cv, m_configure_mutex, m_reader_done);
		m_reader.join();
	}
	if (overlay != nullptr) {
		overlay_destroy(overlay);
		overlay = nullptr;
	}
}

bool lc_session::write_frame(const std::string& payload) {
	const auto    size      = static_cast<uint32_t>(payload.size());
	const uint8_t header[4] = {static_cast<uint8_t>(size), static_cast<uint8_t>(size >> 8u), static_cast<uint8_t>(size >> 16u),
	                           static_cast<uint8_t>(size >> 24u)};
	// One buffer, so one write call per frame
	std::string frame(reinterpret_cast<const char*>(header), sizeof(header));
	frame += payload;

#ifdef _WIN32
	if (m_pipe == nullptr) {
		return false;
	}
	for (size_t written = 0; written < frame.size();) {
		DWORD n = 0;
		if (!pipe_io(m_pipe, true, frame.data() + written, static_cast<DWORD>(frame.size() - written), n)) {
			return false;
		}
		written += n;
	}
	return true;
#else
	if (m_fd < 0) {
		return false;
	}
#ifdef MSG_NOSIGNAL
	constexpr int flags = MSG_NOSIGNAL;
#else
	constexpr int flags = 0;
#endif
	const char* ptr = frame.data();
	size_t      len = frame.size();
	while (len != 0) {
		const ssize_t n = ::send(m_fd, ptr, len, flags);
		if (n < 0 && errno == EINTR) {
			continue;
		}
		if (n <= 0) {
			return false;
		}
		ptr += n;
		len -= static_cast<size_t>(n);
	}
	return true;
#endif
}

bool lc_session::read_frame(std::string& out) {
	const auto read_all = [this](void* data, size_t len) {
		char* ptr = static_cast<char*>(data);
		while (len != 0) {
#ifdef _WIN32
			DWORD n = 0;
			if (!pipe_io(m_pipe, false, ptr, static_cast<DWORD>(len), n)) {
				return false;
			}
#else
			const ssize_t n = ::recv(m_fd, ptr, len, 0);
			if (n < 0 && errno == EINTR) {
				continue;
			}
			if (n <= 0) {
				return false;
			}
#endif
			ptr += n;
			len -= static_cast<size_t>(n);
		}
		return true;
	};

	uint8_t header[4] {};
	if (!read_all(header, sizeof(header))) {
		return false;
	}
	const uint32_t size = static_cast<uint32_t>(header[0]) | (static_cast<uint32_t>(header[1]) << 8u) | (static_cast<uint32_t>(header[2]) << 16u) |
	                      (static_cast<uint32_t>(header[3]) << 24u);
	if (size > MAX_FRAME) {
		log(LC_LOG_ERROR, "frame too big (%u bytes)", size);
		return false;
	}
	out.resize(size);
	return read_all(out.data(), size);
}

void lc_session::dispatch(const std::string& frame) {
	const auto msg = protocol::parse_host(frame);
	if (!msg.ok) {
		log(LC_LOG_ERROR, "bad message: %.200s", frame.c_str());
		return;
	}
	log(LC_LOG_INFO, "received: %s", msg.type.c_str());

	const auto& cb = callbacks;
	if (msg.type == "accepted") {
		std::lock_guard lock(accepted_mutex);
		title = msg.title;
		api   = msg.api;
	} else if (msg.type == "configure") {
		// Boot waits for this, so it's handed over here rather than through a callback
		std::lock_guard lock(m_configure_mutex);
		if (m_configure_taken) {
			log(LC_LOG_ERROR, "configure came after boot started; ignored");
		} else {
			m_configure = msg.values;
			m_configure_cv.notify_all();
		}
	} else if (msg.type == "quit") {
		if (cb.quit != nullptr) {
			cb.quit(cb.user);
		}
	} else if (msg.type == "overlay_opened" || msg.type == "overlay_closed") {
		const bool opened = msg.type == "overlay_opened";
		if (overlay != nullptr) {
			overlay_on_menu(overlay, opened);
		}
		if (cb.overlay != nullptr) {
			cb.overlay(cb.user, opened);
		}
	} else if (msg.type == "overlay_surface") {
		if (overlay != nullptr) {
			overlay_on_surfaces(overlay, msg);
		}
	} else if (msg.type == "overlay_visible") {
		if (overlay != nullptr) {
			overlay_on_visible(overlay, msg.visible);
		}
	} else if (msg.type == "osk_available") {
		osk_set_available(msg.available);
	} else if (msg.type == "osk_edit") {
		if (!msg.edit_ok) {
			log(LC_LOG_ERROR, "osk_edit: an edit it doesn't know; ignored");
		} else if (osk_is_open(msg.osk) && cb.osk_edit != nullptr) {
			lc_osk_edit edit {};
			edit.op     = msg.edit_op;
			edit.text   = msg.edit_text.c_str();
			edit.by     = msg.edit_by;
			edit.cursor = msg.edit_cursor;
			cb.osk_edit(cb.user, msg.osk, msg.seq, &edit);
		}
	} else if (msg.type == "osk_finish") {
		if (osk_is_open(msg.osk) && cb.osk_finish != nullptr) {
			cb.osk_finish(cb.user, msg.osk, msg.accepted);
		}
	} else if (msg.type == "invoke") {
		if (cb.invoke != nullptr) {
			cb.invoke(cb.user, msg.key.c_str(), msg.value ? msg.value->c_str() : nullptr);
		}
	} else if (msg.type == "setting") {
		if (cb.setting != nullptr) {
			cb.setting(cb.user, msg.key.c_str(), msg.value ? msg.value->c_str() : nullptr);
		}
	} else if (msg.type == "scan") {
		if (runner.scan != nullptr) {
			runner.scan(runner.user);
		}
	} else if (msg.type == "launch") {
		if (runner.launch != nullptr) {
			std::vector<const char*> args;
			for (const auto& arg: msg.args) {
				args.push_back(arg.c_str());
			}
			lc_launch_request request {};
			request.launch      = msg.launch;
			request.game        = msg.game;
			request.external_id = msg.external_id.c_str();
			request.game_dir    = msg.game_dir.c_str();
			request.args        = args.data();
			request.arg_count   = args.size();
			runner.launch(runner.user, &request);
		}
	} else if (msg.type == "list_achievements") {
		if (runner.list_achievements != nullptr) {
			runner.list_achievements(runner.user, msg.game, msg.external_id.c_str(), msg.game_dir.c_str());
		}
	}
	// Anything newer is only logged
}

void lc_session::osk_set_available(bool available, bool notify) {
	{
		std::lock_guard lock(m_osk_mutex);
		if (m_osk_available == available) {
			return;
		}
		m_osk_available = available;
		if (!available && m_osk_open != 0) {
			// The emulator shows its own for one that was open (lc_callbacks.osk_available). launcherd
			// is told it's gone, for an open that crossed the offer ending and came after a new one
			send(protocol::osk_closed(m_osk_open, LC_OSK_ABORTED));
			m_osk_open = 0;
		}
	}
	log(LC_LOG_INFO, "on-screen keyboard %s", available ? "available" : "not available");
	if (notify && callbacks.osk_available != nullptr) {
		callbacks.osk_available(callbacks.user, available);
	}
}

bool lc_session::osk_is_open(uint32_t osk) {
	std::lock_guard lock(m_osk_mutex);
	return osk != 0 && m_osk_open == osk;
}

bool lc_session::osk_available() {
	std::lock_guard lock(m_osk_mutex);
	return m_osk_available;
}

uint32_t lc_session::osk_open(const lc_osk_params& params) {
	uint32_t closed = 0;
	uint32_t id     = 0;
	{
		std::lock_guard lock(m_osk_mutex);
		if (!m_osk_available) {
			return 0;
		}
		closed = m_osk_open;
		if (++m_osk_next == 0) {
			m_osk_next = 1;
		}
		id         = m_osk_next;
		m_osk_open = id;
		// Queued under the lock, so a close of the last one goes out before this opens
		if (closed != 0) {
			send(protocol::osk_closed(closed, LC_OSK_ABORTED));
		}
		send(protocol::osk_open(id, params));
	}
	return id;
}

void lc_session::osk_close(uint32_t osk, lc_osk_result result) {
	{
		std::lock_guard lock(m_osk_mutex);
		if (osk == 0 || m_osk_open != osk) {
			// Already closed, or given back with osk_available
			return;
		}
		m_osk_open = 0;
		send(protocol::osk_closed(osk, result));
	}
}

void lc_session::osk_text(uint32_t osk, uint64_t seq, const char* text, uint32_t cursor) {
	std::lock_guard lock(m_osk_mutex);
	if (osk != 0 && m_osk_open == osk) {
		send(protocol::osk_text(osk, seq, text, cursor));
	}
}

void lc_session::reader() {
	std::string frame;
	while (read_frame(frame)) {
		dispatch(frame);
	}
	bool closing = false;
	{
		std::lock_guard lock(m_queue_mutex);
		closing = m_stop;
	}
	// No shell draws its keyboards now; told only if the emulator didn't close the session itself
	osk_set_available(false, !closing);
	if (!closing) {
		log(LC_LOG_INFO, "connection closed");
		if (callbacks.closed != nullptr) {
			callbacks.closed(callbacks.user);
		}
	}
	// Nothing more will come, so boot needn't wait for configure
	std::lock_guard lock(m_configure_mutex);
	m_reader_done = true;
	m_configure_cv.notify_all();
}

bool lc_session::wait_configure(std::chrono::milliseconds timeout, const lc_kv** values, size_t* count) {
	std::unique_lock lock(m_configure_mutex);
	if (!m_configure_cv.wait_until(lock, m_connected_at + timeout, [this]() { return m_configure.has_value() || m_reader_done; })) {
		log(LC_LOG_INFO, "no configure in time; the emulator's own options stay");
	}
	m_configure_taken = true;
	m_configure_kv.clear();
	if (m_configure) {
		for (const auto& [key, value]: *m_configure) {
			m_configure_kv.push_back({key.c_str(), value ? value->c_str() : nullptr});
		}
	}
	if (values != nullptr) {
		*values = m_configure_kv.data();
	}
	if (count != nullptr) {
		*count = m_configure_kv.size();
	}
	return m_configure.has_value();
}

void lc_session::writer() {
	for (;;) {
		Queued item;
		{
			std::unique_lock lock(m_queue_mutex);
			m_queue_cv.wait(lock, [this]() { return !m_queue.empty() || m_stop; });
			if (m_queue.empty()) {
				m_writer_done = true;
				m_flushed_cv.notify_all();
				return;
			}
			item = std::move(m_queue.front());
			m_queue.pop_front();
			m_writing = true;
		}
		// A write fails once launcherd has gone (it may have been restarted); say so once
		if (!write_frame(item.msg.json) && !m_write_failed.exchange(true)) {
			log(LC_LOG_ERROR, "could not send '%s'; later messages are dropped", item.msg.type.c_str());
		}
		{
			std::lock_guard lock(m_queue_mutex);
			m_writing = false;
		}
		m_flushed_cv.notify_all();
	}
}

void lc_session::send(protocol::Message msg) {
	{
		std::lock_guard lock(m_queue_mutex);
		if (m_stop) {
			return;
		}
		// Only frames still waiting are replaced, so updates coalesce only while the queue backs
		// up. Stop at the id's latest frame: an update after activity_done starts a new activity.
		if (msg.type == "activity") {
			for (auto it = m_queue.rbegin(); it != m_queue.rend(); ++it) {
				if (it->msg.activity_id == msg.activity_id) {
					if (it->msg.type == "activity") {
						it->msg.json = std::move(msg.json);
						return;
					}
					break;
				}
			}
		}
		m_queue.push_back({std::move(msg)});
	}
	m_queue_cv.notify_one();
}

bool lc_session::flush(std::chrono::milliseconds timeout) {
	std::unique_lock lock(m_queue_mutex);
	return m_flushed_cv.wait_for(lock, timeout, [this]() { return m_queue.empty() && !m_writing; });
}

void lc_session::ready(uint32_t pid, uint64_t window) {
	std::lock_guard lock(m_view_mutex);
	send(protocol::ready(pid != 0 ? pid : process_id(), window));
	m_ready_sent = true;
	if (m_view) {
		send(protocol::view(m_view->first, m_view->second));
	}
}

void lc_session::view(uint32_t width, uint32_t height) {
	std::lock_guard lock(m_view_mutex);
	if (m_view == std::pair(width, height)) {
		return;
	}
	m_view = std::pair(width, height);
	if (m_ready_sent) {
		log(LC_LOG_INFO, "sending view %ux%u", width, height);
		send(protocol::view(width, height));
	}
}

void lc_session::failed(const char* reason, const char* code) {
	if (m_failed.exchange(true)) {
		return;
	}
	log(LC_LOG_ERROR, "sending failed: %s (%s)", reason != nullptr ? reason : "", code != nullptr ? code : "");
	send(protocol::failed(reason, code));
	if (!flush(std::chrono::seconds(2))) {
		log(LC_LOG_ERROR, "timed out sending failed");
	}
}

void lc_session::activity(const char* id, const char* label, float progress) {
	const std::string key  = id != nullptr ? id : "";
	const std::string text = label != nullptr ? label : "";
	{
		std::lock_guard lock(m_activity_mutex);
		auto&           state = m_activities[key];
		const auto      now   = std::chrono::steady_clock::now();
		if (state.label == text && now - state.last_sent < ACTIVITY_INTERVAL) {
			return;
		}
		state.label     = text;
		state.last_sent = now;
	}
	send(protocol::activity(id, label, progress));
}

void lc_session::activity_done(const char* id) {
	{
		std::lock_guard lock(m_activity_mutex);
		if (m_activities.erase(id != nullptr ? id : "") == 0) {
			return;
		}
	}
	send(protocol::activity_done(id));
}

// The C API

extern "C" {

bool lc_env_present(void) {
	return !get_env(ENV_ENDPOINT).empty() && !get_env(ENV_SESSION).empty();
}

// Both kinds of connection: a game session (`runner` NULL), or a plugin runner
static lc_session* connect_as(const lc_connect_info* info, const lc_callbacks* callbacks, const lc_runner_callbacks* runner,
                              char* error, size_t error_len) {
	std::string why;
	const auto  fail = [&](lc_session* session) -> lc_session* {
        if (session != nullptr) {
            session->log(LC_LOG_ERROR, "%s", why.c_str());
        }
        if (error != nullptr && error_len != 0) {
            std::snprintf(error, error_len, "%s", why.c_str());
        }
        delete session;
        return nullptr;
	};
	lc_connect_info chosen = info != nullptr ? *info : lc_connect_info {};
	if (runner == nullptr) {
		// The runner capabilities would make the hello leave "session" out
		chosen.capabilities &= ~protocol::LC_CAP_RUNNER;
	} else {
		if ((chosen.capabilities & protocol::LC_CAP_RUNNER) == 0) {
			why = "a plugin runner needs LC_CAP_LIBRARY or another runner capability";
			return fail(nullptr);
		}
	}
	if (!lc_env_present()) {
		why = "not started by the launcher (LAUNCHER_ENDPOINT or LAUNCHER_SESSION isn't set)";
		return fail(nullptr);
	}
	if (g_session.load() != nullptr) {
		why = "already connected";
		return fail(nullptr);
	}
	auto* session = new lc_session();
	if (callbacks != nullptr) {
		session->callbacks = *callbacks;
	}
	if (runner != nullptr) {
		session->runner = *runner;
	}
	if (!session->connect(chosen, why)) {
		return fail(session);
	}
	session->start();
	g_session = session;
	return session;
}

lc_session* lc_connect(const lc_connect_info* info, const lc_callbacks* callbacks, char* error, size_t error_len) {
	return connect_as(info, callbacks, nullptr, error, error_len);
}

lc_session* lc_runner_connect(const lc_connect_info* info, const lc_callbacks* callbacks, const lc_runner_callbacks* runner,
                              char* error, size_t error_len) {
	static const lc_runner_callbacks none {};
	return connect_as(info, callbacks, runner != nullptr ? runner : &none, error, error_len);
}

lc_session* lc_get(void) {
	return g_session.load();
}

const char* lc_title(lc_session* session) {
	if (session == nullptr) {
		return "";
	}
	// Set once, by the first message after the welcome
	std::lock_guard lock(session->accepted_mutex);
	return session->title.c_str();
}

int lc_api(lc_session* session) {
	if (session == nullptr) {
		return 0;
	}
	std::lock_guard lock(session->accepted_mutex);
	return session->api;
}

bool lc_wait_configure(lc_session* session, uint32_t timeout_ms, const lc_kv** values, size_t* count) {
	if (session == nullptr) {
		if (values != nullptr) {
			*values = nullptr;
		}
		if (count != nullptr) {
			*count = 0;
		}
		return false;
	}
	return session->wait_configure(std::chrono::milliseconds(timeout_ms), values, count);
}

void lc_booting(lc_session* session) {
	if (session != nullptr) {
		session->send(protocol::booting());
	}
}

void lc_ready(lc_session* session, uint32_t pid, uint64_t window) {
	if (session != nullptr) {
		session->ready(pid, window);
	}
}

void lc_failed(lc_session* session, const char* reason, const char* code) {
	if (session != nullptr) {
		session->failed(reason, code);
	}
}

void lc_paused(lc_session* session, bool paused) {
	if (session != nullptr) {
		session->send(protocol::paused(paused));
	}
}

void lc_activity(lc_session* session, const char* id, const char* label, float progress) {
	if (session != nullptr) {
		session->activity(id, label, progress);
	}
}

void lc_activity_done(lc_session* session, const char* id) {
	if (session != nullptr) {
		session->activity_done(id);
	}
}

void lc_presence(lc_session* session, const char* text) {
	if (session != nullptr) {
		session->send(protocol::presence(text));
	}
}

void lc_notify(lc_session* session, const char* title, const char* body, bool error) {
	if (session != nullptr) {
		session->send(protocol::notify(title, body, error));
	}
}

void lc_achievements(lc_session* session, const lc_achievement* list, size_t count) {
	if (session != nullptr) {
		session->log(LC_LOG_INFO, "sending %zu achievements", count);
		session->send(protocol::achievements(list, count));
	}
}

void lc_unlocked(lc_session* session, const char* id, int64_t at) {
	if (session != nullptr) {
		session->log(LC_LOG_INFO, "sending unlocked %s", id != nullptr ? id : "");
		session->send(protocol::unlocked(id, at));
	}
}

void lc_achievement_progress(lc_session* session, const char* id, uint32_t current, uint32_t total) {
	if (session != nullptr) {
		session->send(protocol::achievement_progress(id, current, total));
	}
}

void lc_overlay_items(lc_session* session, const lc_setting* items, size_t count) {
	if (session != nullptr) {
		session->send(protocol::overlay_items(items, count));
	}
}

bool lc_osk_available(lc_session* session) {
	return session != nullptr && session->osk_available();
}

uint32_t lc_osk_open(lc_session* session, const lc_osk_params* params) {
	return session != nullptr && params != nullptr ? session->osk_open(*params) : 0;
}

void lc_osk_text(lc_session* session, uint32_t osk, uint64_t seq, const char* text, uint32_t cursor) {
	if (session != nullptr) {
		session->osk_text(osk, seq, text, cursor);
	}
}

void lc_osk_close(lc_session* session, uint32_t osk, lc_osk_result result) {
	if (session != nullptr) {
		session->osk_close(osk, result);
	}
}

bool lc_flush(lc_session* session, uint32_t timeout_ms) {
	return session == nullptr || session->flush(std::chrono::milliseconds(timeout_ms));
}

void lc_close(lc_session* session) {
	if (session == nullptr) {
		return;
	}
	lc_session* expected = session;
	g_session.compare_exchange_strong(expected, nullptr);
	session->close();
	delete session;
}

} // extern "C"
