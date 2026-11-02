#include "internal.h"
#include "transport.h"

#include <cstring>

#ifndef _WIN32
#include <unistd.h>
#endif

using namespace lc;

namespace {

// Presents still needed after a change, so the screen catches up even with no game frames
// (hiding the overlay, say)
constexpr uint32_t PRESENTS_AFTER_CHANGE = 3;
// How many presents the last frame of the old surfaces stays up waiting for the shell's first in new
// ones (a resize), before the new ones are taken anyway
constexpr uint32_t PENDING_WAIT_PRESENTS = 120;

struct SurfaceSet {
	std::vector<std::unique_ptr<lc_surface>> buffers;
	std::unique_ptr<lc_surface>              header;
	uint32_t                                 width  = 0;
	uint32_t                                 height = 0;
	bool                                     lists  = false; // The buffers are draw-list slots
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
	lc_overlay(lc_session& s, bool lists): session(s), draws_lists(lists) {}

	lc_session& session;
	// LC_CAP_DRAW_LISTS: the shell sends draw lists, never buffers, and wants no capture
	const bool draws_lists;

	// Held while the service or its surfaces change, and while the capture goes out, so a capture
	// never goes to a service whose surfaces the game isn't using yet
	std::mutex         io_mutex;
	transport::Service service;

	// Shared between the reader, the present thread and the sender
	std::mutex                mutex;
	std::condition_variable   cv; // For the sender
	std::optional<SurfaceSet> pending;
	uint32_t                  pending_serial = 0; // Counts the sets that came to be `pending`
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
	std::atomic<bool>     input {false}; // Shown, and the game's keys and mouse are the shell's
	std::atomic<uint32_t> presents_owed {0};

	// The present thread's own
	SurfaceSet                     current;
	std::vector<const lc_surface*> current_ptrs;
	transport::HeaderMap           header;
	// The pending set's header, to see when the shell has drawn into it, and which set that is
	transport::HeaderMap pending_header;
	uint32_t             pending_header_serial = 0;
	uint32_t             pending_waited        = 0;
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
	// Draw lists: the current set's slots, mapped, and the list from the slot the game holds,
	// copied out of it (so the shell can't change it under the emulator) and checked
	std::vector<std::unique_ptr<transport::SlotMap>> slots;
	std::vector<uint64_t>                            list_bytes;
	lc_list                                          list {};
	bool                                             has_list   = false;
	bool                                             list_unread = false; // `mine` is a list not yet read
	bool                                             list_fresh = false; // This present's list is new

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

lc_overlay* overlay_create(lc_session& session, bool lists) {
	auto* overlay   = new lc_overlay(session, lists);
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
	} else if (msg.lists != overlay->draws_lists) {
		overlay->log(LC_LOG_ERROR, "overlay_surface: the launcher sent %s to a session that draws %s",
		             msg.lists ? "draw lists" : "buffers", overlay->draws_lists ? "draw lists" : "buffers");
		overlay->service.disconnect();
	} else if (msg.service.empty()) {
		overlay->log(LC_LOG_ERROR, "overlay_surface: no service to get the surfaces from");
	} else if (!overlay->service.connect(msg.service, error)) {
		overlay->log(LC_LOG_ERROR, "overlay_surface: %s", error.c_str());
	} else if (auto surfaces = overlay->service.fetch(msg.buffers, msg.header, msg.width, msg.height, msg.lists, error)) {
		set.buffers = std::move(surfaces->buffers);
		set.header  = std::move(surfaces->header);
		set.width   = msg.width;
		set.height  = msg.height;
		set.lists   = msg.lists;
		overlay->log(LC_LOG_INFO, "overlay: %zu %s for %ux%u, header %u", set.buffers.size(),
		             msg.lists ? "draw-list slot(s)" : "buffer(s)", set.width, set.height, msg.header);
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
		overlay->pending_serial++;
		overlay->ready        = false;
		overlay->drop_service = false;
	}
	overlay->presents_owed = PRESENTS_AFTER_CHANGE;
}

void overlay_on_visible(lc_overlay* overlay, bool visible, bool input) {
	overlay->input = visible && input;
	if (overlay->visible.exchange(visible) != visible) {
		overlay->presents_owed = PRESENTS_AFTER_CHANGE;
	}
}

void overlay_on_menu(lc_overlay* overlay, bool opened) {
	std::lock_guard lock(overlay->mutex);
	overlay->drop_capture();
	overlay->menu++;
	// The next present copies the frame it paused on for the shell; one drawing lists draws the
	// menu over its live frame, so the shell needs none
	overlay->capture_requested = opened && !overlay->draws_lists;
}

// Maps the current set's draw-list slots. Under the mutex, on the present thread.
bool map_slots(lc_overlay& o) {
	for (const auto& buffer: o.current.buffers) {
		auto slot = std::make_unique<transport::SlotMap>();
		if (!slot->map(*buffer)) {
			o.log(LC_LOG_ERROR, "overlay: draw-list slot %u is missing or not sealed", buffer->id);
			o.slots.clear();
			return false;
		}
		o.slots.push_back(std::move(slot));
	}
	return !o.slots.empty();
}

// Draw lists that can't be drawn: the overlay lets the shell's service go, as for surfaces the
// emulator couldn't import, so the shell comes to the front instead of leaving a menu nobody
// sees with the input. Without the mutex.
void lists_failed(lc_overlay& o) {
	o.broken   = true;
	o.has_list = false;
	{
		std::lock_guard lock(o.mutex);
		o.drop_capture();
		o.drop_service = !o.pending;
	}
	o.cv.notify_all();
}

bool overlay_draws_lists(lc_overlay* overlay) {
	return overlay->draws_lists;
}

void overlay_lists_failed(lc_overlay* overlay) {
	if (!overlay->broken) {
		overlay->log(LC_LOG_ERROR, "overlay: the emulator can't draw the launcher's menu; it comes to the front instead");
		lists_failed(*overlay);
	}
}

void overlay_gpu_times(lc_overlay* overlay, float render, float glass, float draw, float composite) {
	auto& o = *overlay;
	if (!o.current.lists || !o.header.mapped()) {
		return;
	}
	auto* words = o.header.lock();
	if (words == nullptr) {
		return;
	}
	// A sequence lock: the count is odd while the times are written, so the shell reads all four
	// from one present or tries again
	auto&       count  = words[transport::HeaderMap::GPU_COUNT];
	const float times[4] = {render, glass, draw, composite};
	count.fetch_add(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	for (size_t i = 0; i < 4; i++) {
		uint32_t bits = 0;
		std::memcpy(&bits, &times[i], sizeof(bits));
		words[transport::HeaderMap::GPU_TIMES + i].store(bits, std::memory_order_relaxed);
	}
	count.fetch_add(1, std::memory_order_release);
	o.header.unlock();
}

// Copies the list in the slot the game holds out of it, and checks it. False if the slot can't be
// read just now (it's read again next present); a list that doesn't read breaks the overlay.
bool read_list(lc_overlay& o) {
	size_t         size  = 0;
	const uint8_t* bytes = o.slots[o.mine]->lock(size);
	if (bytes == nullptr) {
		return false;
	}
	// The bytes the shell says it used, if the slot holds that many
	uint32_t used = 0;
	if (size >= 12) {
		std::memcpy(&used, bytes + 8, 4);
	}
	const bool fits = used != 0 && used <= size;
	if (fits) {
		o.list_bytes.resize((used + 7) / 8);
		std::memcpy(o.list_bytes.data(), bytes, used);
	}
	o.slots[o.mine]->unlock();
	if (!fits || !lc_list_read(o.list_bytes.data(), used, &o.list)) {
		o.log(LC_LOG_ERROR, "overlay: the launcher sent a draw list this library can't read; the menu isn't drawn");
		lists_failed(o);
		return true;
	}
	o.has_list = true;
	// So a new set of slots waits for its first list before replacing this one (pending_drawn)
	o.has_content = true;
	return true;
}

// The draw-list side of lc_overlay_begin: takes the shell's newest list, if it has one, and hands
// over the list the game holds. Each list is taken whole, and read before another is taken, so
// the emulator sees every list's changes once, in order.
void begin_lists(lc_overlay& o, bool again, lc_frame* frame) {
	if (!again && o.visible && !o.broken && o.header.mapped()) {
		o.list_fresh = false;
		if (!o.list_unread) {
			if (auto* words = o.header.lock(); words != nullptr) {
				const bool newer = words[0].load(std::memory_order_acquire) == transport::HeaderMap::MAGIC &&
				                   (words[1].load(std::memory_order_acquire) & transport::HeaderMap::NEWER) != 0;
				if (newer) {
					// Nothing reads the game's slot once its list is copied out: give it straight back
					const uint32_t got = words[1].exchange(o.mine, std::memory_order_acq_rel) & ~transport::HeaderMap::NEWER;
					if (got >= o.slots.size()) {
						o.log(LC_LOG_ERROR, "overlay: the launcher published slot %u of %zu", got, o.slots.size());
						o.header.unlock();
						lists_failed(o);
						return;
					} else {
						o.mine        = got;
						o.list_unread = true;
					}
				}
				o.header.unlock();
			}
		}
		if (o.list_unread && !o.broken && read_list(o)) {
			o.list_unread = false;
			o.list_fresh  = o.has_list;
		}
	}
	if (o.visible && !o.broken && o.has_list) {
		frame->list       = &o.list;
		frame->list_fresh = o.list_fresh;
	}
}

// Whether to take the pending set now. While the menu shows, its last frame stays up until the
// shell has drawn into the new set, so a resize (new surfaces at each size) doesn't blank it; but
// not for long, nor when there's nothing to keep showing. Under the mutex, on the present thread.
bool pending_drawn(lc_overlay& o, bool again) {
	const SurfaceSet& next = *o.pending;
	if (!o.visible || !o.has_content || o.broken || o.current.buffers.empty() || next.buffers.empty() ||
	    next.header == nullptr) {
		return true;
	}
	if (o.pending_header_serial != o.pending_serial) {
		o.pending_header.unmap();
		o.pending_header_serial = o.pending_serial;
		o.pending_waited        = 0;
	}
	if (!again && ++o.pending_waited > PENDING_WAIT_PRESENTS) {
		return true;
	}
	if (!o.pending_header.mapped() && !o.pending_header.map(*next.header)) {
		return true;
	}
	auto* words = o.pending_header.lock();
	if (words == nullptr) {
		return false;
	}
	const bool drawn = words[0].load(std::memory_order_acquire) == transport::HeaderMap::MAGIC &&
	                   (words[1].load(std::memory_order_acquire) & transport::HeaderMap::NEWER) != 0;
	o.pending_header.unlock();
	return drawn;
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
		if (o.pending && !pending_drawn(o, again)) {
			// The menu keeps its last frame (stretched to the window) until there's one in the new set
		} else if (o.pending) {
			o.pending_header.unmap();
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
			o.slots.clear();
			o.has_list    = false;
			o.list_unread = false;
			o.list_fresh  = false;
			if (o.current.lists) {
				o.broken = o.broken || o.current.header == nullptr || !map_slots(o);
				// Nothing for the emulator to import, so nobody else says they can't be used
				if (o.broken) {
					o.drop_service = true;
					wake           = true;
				}
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
	frame->buffer_count = o.broken || o.current.lists ? 0 : static_cast<uint32_t>(o.current_ptrs.size());
	frame->width        = o.current.width;
	frame->height       = o.current.height;

	const auto count = frame->buffer_count;
	if (o.current.lists) {
		begin_lists(o, again, frame);
	} else if (o.visible && count != 0) {
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

bool lc_overlay_wants_input(lc_overlay* overlay) {
	return overlay != nullptr && overlay->input;
}

void lc_overlay_input(lc_overlay* overlay, const lc_input* input) {
	if (overlay == nullptr || input == nullptr || !overlay->input) {
		return;
	}
	auto msg = lc::protocol::overlay_input(*input);
	if (!msg.type.empty()) {
		overlay->session.send(std::move(msg));
	}
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
