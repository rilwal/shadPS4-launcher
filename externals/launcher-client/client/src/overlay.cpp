#include "internal.h"
#include "transport.h"

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace lc;

namespace {

// Presents still needed after a change, so the screen catches up even with no game frames
// (hiding the overlay, say)
constexpr uint32_t PRESENTS_AFTER_CHANGE = 3;

struct SurfaceSet {
	std::vector<std::unique_ptr<lc_surface>> buffers;
	std::unique_ptr<lc_surface>              header;
	uint32_t                                 width  = 0;
	uint32_t                                 height = 0;
};

// A capture the emulator handed over, until it's sent
struct Capture {
	lc_capture info {};
	uint32_t   id     = 0;
	uint64_t   serial = 0;
	uint32_t   menu   = 0; // The menu it was asked for (lc_overlay::menu)

	Capture() = default;
	Capture(const Capture&)            = delete;
	Capture& operator=(const Capture&) = delete;
	~Capture() { transport::release_capture(info); }
};

} // namespace

struct lc_overlay {
	explicit lc_overlay(lc_session& s): session(s) {}

	lc_session& session;

	// Held while the service or its surfaces change, and while the capture goes out, so a capture
	// never goes to a service whose surfaces the game isn't using yet
	std::mutex         io_mutex;
	transport::Service service;

	// Shared between the reader, the present thread and the sender
	std::mutex                mutex;
	std::condition_variable   cv; // For the sender
	std::optional<SurfaceSet> pending;
	std::unique_ptr<Capture>  waiting; // Copied, until the GPU finishes it
	std::unique_ptr<Capture>  ready_capture; // Finished, until it's sent
	bool                      ready        = false; // The current set is imported and nothing newer came
	bool                      drop_service = false; // The emulator couldn't use the surfaces: let them go
	bool                      stop         = false;
	// The capture's state, which the menu opening or closing resets: each opening is a new `menu`,
	// and a capture asked for under an older one is dropped
	uint32_t                  menu              = 0;
	bool                      capture_requested = false;
	bool                      capture_in_flight = false; // Asked for and not yet sent or dropped
	std::thread               sender;

	std::atomic<bool>     visible {false};
	std::atomic<uint32_t> presents_owed {0};

	// The present thread's own
	SurfaceSet                     current;
	std::vector<const lc_surface*> current_ptrs;
	transport::HeaderMap           header;
	uint32_t                       generation   = 0;
	uint32_t                       mine         = 2; // The buffer the game owns: buffer 2, empty
	bool                           has_content  = false; // The game has had a buffer from the shell
	bool                           broken       = false; // The shell broke the exchange, or the import failed
	uint64_t                       sampled      = 0; // The serial that last read `mine`, 0 if none
	bool                           unseen       = false; // `mine` hasn't been drawn since the shell wrote it
	int32_t                        drawn        = -1; // What this present was told to draw
	bool                           in_present   = false; // Between lc_overlay_begin and lc_overlay_end
	bool                           capturing    = false; // This present was asked for the capture
	uint32_t                       capture_menu = 0; // ...for this menu

	void log(lc_log_level level, const char* format, ...) LC_PRINTF(3, 4) {
		va_list args;
		va_start(args, format);
		vlog(session.callbacks, level, format, args);
		va_end(args);
	}

	// With `mutex` held: forget any capture, as the menu opens or closes or the surfaces go
	void drop_capture() {
		waiting.reset();
		ready_capture.reset();
		capture_in_flight = false;
	}

	void send_capture();
	void let_service_go();
	void run_sender();
};

void lc_overlay::send_capture() {
	std::lock_guard          io(io_mutex);
	std::unique_ptr<Capture> capture;
	{
		std::lock_guard lock(mutex);
		if (ready_capture == nullptr || !ready || pending || !service.connected()) {
			return;
		}
		capture = std::move(ready_capture);
	}
	std::string error;
	if (!service.send_capture(capture->info, capture->id, error)) {
		log(LC_LOG_ERROR, "couldn't hand the capture to the launcher: %s", error.c_str());
	} else {
		log(LC_LOG_INFO, "sending capture_surface %u (%ux%u)", capture->id, capture->info.width, capture->info.height);
		session.send(protocol::capture_surface(capture->id, capture->info.width, capture->info.height));
	}
	std::lock_guard lock(mutex);
	if (capture->menu == menu) {
		capture_in_flight = false;
	}
}

void lc_overlay::let_service_go() {
	std::lock_guard io(io_mutex);
	std::lock_guard lock(mutex);
	// A newer set may have come meanwhile, from a service of its own
	if (drop_service && !pending) {
		service.disconnect();
	}
	drop_service = false;
}

void lc_overlay::run_sender() {
	for (;;) {
		bool drop = false;
		{
			std::unique_lock lock(mutex);
			cv.wait(lock, [this]() { return stop || drop_service || (ready_capture != nullptr && ready && !pending); });
			if (stop) {
				return;
			}
			drop = drop_service;
		}
		if (drop) {
			let_service_go();
		} else {
			send_capture();
		}
	}
}

namespace lc {

lc_overlay* overlay_create(lc_session& session) {
	auto* overlay   = new lc_overlay(session);
	overlay->sender = std::thread([overlay]() { overlay->run_sender(); });
	return overlay;
}

lc_session* overlay_session(lc_overlay* overlay) {
	return &overlay->session;
}

void overlay_stop(lc_overlay* overlay) {
	{
		std::lock_guard lock(overlay->mutex);
		overlay->stop = true;
	}
	overlay->cv.notify_all();
	if (overlay->sender.joinable()) {
		overlay->sender.join();
	}
	std::lock_guard io(overlay->io_mutex);
	overlay->service.disconnect();
}

void overlay_destroy(lc_overlay* overlay) {
	overlay_stop(overlay);
	delete overlay;
}

void overlay_on_surfaces(lc_overlay* overlay, const protocol::HostMessage& msg) {
	std::lock_guard io(overlay->io_mutex);
	{
		// The session is closing: don't connect to a service again
		std::lock_guard lock(overlay->mutex);
		if (overlay->stop) {
			return;
		}
	}

	// No buffers takes the overlay away. The surfaces come from the shell's service; the ids say
	// which they should be.
	SurfaceSet  set;
	std::string error;
	if (msg.buffers.empty()) {
		overlay->service.disconnect();
	} else if (msg.service.empty()) {
		overlay->log(LC_LOG_ERROR, "overlay_surface: no service to get the surfaces from");
	} else if (!overlay->service.connect(msg.service, error)) {
		overlay->log(LC_LOG_ERROR, "overlay_surface: %s", error.c_str());
	} else if (auto surfaces = overlay->service.fetch(msg.buffers, msg.header, msg.width, msg.height, error)) {
		set.buffers = std::move(surfaces->buffers);
		set.header  = std::move(surfaces->header);
		set.width   = msg.width;
		set.height  = msg.height;
		overlay->log(LC_LOG_INFO, "overlay: %zu buffer(s) of %ux%u, header %u", set.buffers.size(), set.width, set.height,
		             msg.header);
	} else {
		overlay->log(LC_LOG_ERROR, "overlay_surface: %s", error.c_str());
		// No capture goes out without the surfaces, so the shell comes to the front instead
		overlay->service.disconnect();
	}

	{
		std::lock_guard lock(overlay->mutex);
		if (set.buffers.empty()) {
			// The menu is closed, or can't be drawn: a capture isn't wanted
			overlay->drop_capture();
		}
		overlay->pending      = std::move(set);
		overlay->ready        = false;
		overlay->drop_service = false;
	}
	overlay->presents_owed = PRESENTS_AFTER_CHANGE;
}

void overlay_on_visible(lc_overlay* overlay, bool visible) {
	if (overlay->visible.exchange(visible) != visible) {
		overlay->presents_owed = PRESENTS_AFTER_CHANGE;
	}
}

void overlay_on_menu(lc_overlay* overlay, bool opened) {
	std::lock_guard lock(overlay->mutex);
	overlay->drop_capture();
	overlay->menu++;
	// The next present copies the frame it paused on for the shell
	overlay->capture_requested = opened;
}

} // namespace lc

extern "C" {

bool lc_overlay_supported(void) {
	return transport::supported();
}

lc_overlay* lc_overlay_get(lc_session* session) {
	return session != nullptr ? session->overlay : nullptr;
}

void lc_overlay_view(lc_overlay* overlay, uint32_t width, uint32_t height) {
	if (overlay != nullptr && width != 0 && height != 0) {
		overlay->session.view(width, height);
	}
}

void lc_overlay_begin(lc_overlay* overlay, uint64_t completed, lc_frame* frame) {
	if (frame == nullptr) {
		return;
	}
	*frame      = {};
	frame->draw = -1;
	if (overlay == nullptr) {
		return;
	}
	auto& o = *overlay;

	// lc_overlay_begin again in the same present (after waiting for frame->wait) takes nothing
	// that's once a present a second time
	const bool again = o.in_present;
	o.in_present     = true;

	bool wake = false;
	{
		std::lock_guard lock(o.mutex);
		if (o.pending) {
			// The old set's surfaces close here; the emulator's imports hold their own
			o.current = std::move(*o.pending);
			o.pending.reset();
			o.current_ptrs.clear();
			for (const auto& buffer: o.current.buffers) {
				o.current_ptrs.push_back(buffer.get());
			}
			o.header.unmap();
			o.generation++;
			o.mine        = 2;
			o.has_content = false;
			o.sampled     = 0;
			o.unseen      = false;
			o.broken      = false;
			o.ready       = false;
			if (o.current.header != nullptr && !o.header.map(*o.current.header)) {
				o.log(LC_LOG_ERROR, "overlay: header %u is missing, too small or not sealed", o.current.header->id);
				o.broken = true;
			}
		}
		if (o.waiting != nullptr && completed >= o.waiting->serial) {
			o.ready_capture = std::move(o.waiting);
			wake            = true;
		}
		if (!again) {
			o.capturing = o.capture_requested;
			if (o.capture_requested) {
				// In flight before the request clears, so lc_overlay_wants_presents stays true throughout
				o.capture_requested = false;
				o.capture_in_flight = true;
				o.capture_menu      = o.menu;
			}
		}
	}
	if (wake) {
		o.cv.notify_all();
	}

	if (uint32_t owed = o.presents_owed.load(); !again && owed != 0) {
		o.presents_owed.compare_exchange_strong(owed, owed - 1);
	}
	frame->capture = o.capturing;

	frame->generation   = o.generation;
	frame->buffers      = o.current_ptrs.data();
	frame->buffer_count = o.broken ? 0 : static_cast<uint32_t>(o.current_ptrs.size());
	frame->width        = o.current.width;
	frame->height       = o.current.height;

	const auto count = frame->buffer_count;
	if (o.visible && count != 0) {
		if (!o.header.mapped()) {
			// Without a header the shell may redraw the one buffer at any time
			frame->draw  = 0;
			frame->fresh = true;
		} else if (auto* words = o.header.lock(); words == nullptr) {
			// Not just now (an IOSurface that won't lock): the buffer it has
			if (!o.broken && o.has_content) {
				frame->draw  = static_cast<int32_t>(o.mine);
				frame->fresh = o.unseen;
			}
		} else {
			const bool newer = words[0].load(std::memory_order_acquire) == transport::HeaderMap::MAGIC &&
			                   (words[1].load(std::memory_order_acquire) & transport::HeaderMap::NEWER) != 0;
			if (newer && o.sampled != 0 && completed < o.sampled) {
				// The shell has a newer frame, but the buffer to give back is still being read. With
				// frames in flight that's so at every present, so it has to be waited for.
				frame->wait = o.sampled;
			} else if (newer) {
				// Give the game's back (bit clear) and take the newer one
				const uint32_t got = words[1].exchange(o.mine, std::memory_order_acq_rel) & ~transport::HeaderMap::NEWER;
				if (got >= count) {
					// The game's buffer is back in the word, so it's no longer the game's to draw
					o.log(LC_LOG_ERROR, "overlay: the launcher published buffer %u of %u", got, count);
					o.broken             = true;
					frame->buffer_count  = 0;
				} else {
					o.mine        = got;
					o.has_content = true;
					o.sampled     = 0;
					o.unseen      = true;
				}
			}
			o.header.unlock();
			if (!o.broken && o.has_content) {
				frame->draw  = static_cast<int32_t>(o.mine);
				frame->fresh = o.unseen;
			}
		}
	}
	o.drawn = frame->draw;
}

void lc_overlay_surfaces_ready(lc_overlay* overlay, uint32_t generation, bool ok) {
	if (overlay == nullptr) {
		return;
	}
	auto& o = *overlay;
	if (generation != o.generation) {
		return;
	}
	{
		std::lock_guard lock(o.mutex);
		if (ok) {
			o.ready = !o.pending && !o.broken && !o.current.buffers.empty();
		} else {
			o.log(LC_LOG_ERROR, "overlay: the emulator couldn't use the launcher's surfaces");
			o.broken = true;
			// No capture goes out, and the sender lets the surfaces' service go (not here, where
			// the service may be busy), so the shell comes to the front
			o.drop_capture();
			o.drop_service = !o.pending;
		}
	}
	o.cv.notify_all();
}

void lc_overlay_end(lc_overlay* overlay, uint64_t serial, bool drew) {
	if (overlay == nullptr) {
		return;
	}
	if (drew && overlay->drawn >= 0) {
		overlay->sampled = serial;
		overlay->unseen  = false;
	}
	overlay->in_present = false;
	overlay->capturing  = false;
}

void lc_overlay_capture(lc_overlay* overlay, const lc_capture* capture, uint64_t serial) {
	if (overlay == nullptr) {
		// Linux's descriptor is the library's to close; macOS's IOSurface isn't retained yet
#if defined(__linux__)
		if (capture != nullptr && capture->fd >= 0) {
			::close(capture->fd);
		}
#endif
		return;
	}
	auto& o = *overlay;
	std::unique_ptr<Capture> c;
	if (capture != nullptr) {
		c         = std::make_unique<Capture>();
		c->info   = *capture;
		c->serial = serial;
		c->menu   = o.capture_menu;
		c->id     = transport::capture_id(c->info);
		transport::adopt_capture(c->info);
	}
	std::lock_guard lock(o.mutex);
	if (!o.capture_in_flight || o.capture_menu != o.menu) {
		// The menu closed (or opened again) since this was asked for; the capture closes here
		return;
	}
	if (c == nullptr) {
		o.log(LC_LOG_ERROR, "overlay: the emulator couldn't copy its frame for the launcher");
		o.capture_in_flight = false;
		return;
	}
	o.waiting = std::move(c);
}

bool lc_overlay_wants_presents(lc_overlay* overlay) {
	if (overlay == nullptr) {
		return false;
	}
	if (overlay->visible || overlay->presents_owed != 0) {
		return true;
	}
	std::lock_guard lock(overlay->mutex);
	return overlay->capture_requested || overlay->capture_in_flight;
}

uint32_t lc_surface_id(const lc_surface* surface) {
	return surface != nullptr ? surface->id : 0;
}

uint32_t lc_surface_stride(const lc_surface* surface) {
	return surface != nullptr ? surface->stride : 0;
}

uint64_t lc_surface_offset(const lc_surface* surface) {
	return surface != nullptr ? surface->offset : 0;
}

uint64_t lc_surface_modifier(const lc_surface* surface) {
	return surface != nullptr ? surface->modifier : 0;
}

int lc_surface_fd(const lc_surface* surface) {
	return surface != nullptr ? surface->fd : -1;
}

void* lc_surface_iosurface(const lc_surface* surface) {
	return surface != nullptr ? surface->iosurface : nullptr;
}

} // extern "C"
