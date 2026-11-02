// What the library's parts share: the session object behind lc_session, and the overlay's hooks
// into it.
#ifndef LAUNCHER_CLIENT_INTERNAL_H
#define LAUNCHER_CLIENT_INTERNAL_H

#include "launcher_client.h"
#include "launcher_overlay.h"
#include "launcher_runner.h"
#include "protocol.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lc {

#if defined(__GNUC__) || defined(__clang__)
#define LC_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define LC_PRINTF(fmt, args)
#endif

// printf-style, to the emulator's log callback or stderr. Any thread.
void vlog(const lc_callbacks& callbacks, lc_log_level level, const char* format, va_list args);

} // namespace lc

struct lc_session {
	lc_session() = default;
	lc_session(const lc_session&)            = delete;
	lc_session& operator=(const lc_session&) = delete;
	~lc_session();

	bool connect(const lc_connect_info& info, std::string& error);
	void start();
	void close();

	// Queues the message for the writer thread. An activity update still waiting is replaced by a
	// newer one for the same id.
	void send(lc::protocol::Message msg);
	bool flush(std::chrono::milliseconds timeout);

	void ready(uint32_t pid, uint64_t window);
	void view(uint32_t width, uint32_t height);
	void failed(const char* reason, const char* code);
	void activity(const char* id, const char* label, float progress);
	void activity_done(const char* id);
	bool wait_configure(std::chrono::milliseconds timeout, const lc_kv** values, size_t* count);

	// The on-screen keyboard (lc_osk_*): one open at a time, by number
	bool     osk_available();
	uint32_t osk_open(const lc_osk_params& params);
	void     osk_text(uint32_t osk, uint64_t seq, const char* text, uint32_t cursor);
	void     osk_close(uint32_t osk, lc_osk_result result);

	void log(lc_log_level level, const char* format, ...) LC_PRINTF(3, 4);

	lc_callbacks                 callbacks {};
	lc_runner_callbacks          runner {}; // A plugin runner's (launcher_runner.h)
	lc_overlay*                  overlay = nullptr;

	// From accepted
	std::mutex  accepted_mutex;
	std::string title;
	int         api = 0;

private:
	bool open_endpoint(const std::string& endpoint, std::string& error);
	bool wait_readable(int timeout_ms);
	bool write_frame(const std::string& payload);
	bool read_frame(std::string& out);
	void shutdown_io();
	void reader();
	void writer();
	void dispatch(const std::string& frame);
	void osk_set_available(bool available, bool notify = true);
	bool osk_is_open(uint32_t osk);

#ifdef _WIN32
	void* m_pipe = nullptr; // The pipe's HANDLE, opened for overlapped I/O
#else
	int m_fd = -1;
#endif
	std::thread m_reader;
	std::thread m_writer;

	struct Queued {
		lc::protocol::Message msg;
	};
	std::mutex              m_queue_mutex;
	std::condition_variable m_queue_cv;   // Something was queued, or stop
	std::condition_variable m_flushed_cv; // The queue emptied, or the writer finished
	std::deque<Queued>      m_queue;
	bool                    m_writing     = false; // The writer took a frame and is writing it
	bool                    m_stop        = false;
	bool                    m_writer_done = false;
	std::atomic<bool>       m_write_failed {false};
	std::atomic<bool>       m_failed {false};

	std::chrono::steady_clock::time_point m_connected_at;
	std::mutex                            m_configure_mutex;
	std::condition_variable               m_configure_cv;
	std::optional<lc::protocol::Values>   m_configure;
	std::vector<lc_kv>                    m_configure_kv; // Pointing into m_configure
	bool                                  m_configure_taken = false;
	bool                                  m_reader_done     = false;

	struct ActivityState {
		std::string                           label;
		std::chrono::steady_clock::time_point last_sent;
	};
	std::mutex                           m_activity_mutex;
	std::map<std::string, ActivityState> m_activities;

	std::mutex                                   m_view_mutex;
	std::optional<std::pair<uint32_t, uint32_t>> m_view;
	bool                                         m_ready_sent = false;

	std::mutex m_osk_mutex;
	bool       m_osk_available = false;
	uint32_t   m_osk_open      = 0; // The keyboard open on the shell, 0 for none
	uint32_t   m_osk_next      = 0;
};

namespace lc {

// The composited overlay (overlay.cpp). The session hands it the overlay messages on its reader
// thread; the present thread calls the lc_overlay_* functions.
// `lists`: the session draws the menu from draw lists (LC_CAP_DRAW_LISTS)
lc_overlay* overlay_create(lc_session& session, bool lists);
void        overlay_destroy(lc_overlay* overlay);
void        overlay_on_surfaces(lc_overlay* overlay, const protocol::HostMessage& msg);
void        overlay_on_visible(lc_overlay* overlay, bool visible);
void        overlay_on_menu(lc_overlay* overlay, bool opened);
lc_session* overlay_session(lc_overlay* overlay);
// Whether it draws the menu from draw lists (LC_CAP_DRAW_LISTS), not the shell's buffers
bool overlay_draws_lists(lc_overlay* overlay);
// The emulator can't draw the lists it's handed: the shell comes to the front instead
void overlay_lists_failed(lc_overlay* overlay);
// The GPU time the game's drawing of the menu took, in milliseconds, for the shell's frame timings:
// written into the header (transport::HeaderMap). On the presenting thread.
void overlay_gpu_times(lc_overlay* overlay, float render, float glass, float draw, float composite);
// Stops its thread and lets the service go, before the session closes its connection.
void overlay_stop(lc_overlay* overlay);

} // namespace lc

#endif // LAUNCHER_CLIENT_INTERNAL_H
