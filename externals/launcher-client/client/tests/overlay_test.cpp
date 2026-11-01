// The composited overlay against a fake launcherd and a fake shell service (Linux): surfaces are
// fetched and checked, the header's buffer trade waits for the GPU, and the capture goes out only
// once the surfaces are in use, before capture_surface names it.
#include "check.h"
#include "fake_launcher.h"
#include "fake_shell.h"
#include "launcher_overlay.h"

#include <atomic>
#include <cstddef>
#include <sys/mman.h>

namespace {

// Presents (drawing nothing) until `done` or a second passes
template <typename F>
bool present_until(lc_overlay* overlay, uint64_t completed, lc_frame& frame, F done) {
	for (int i = 0; i < 100; i++) {
		lc_overlay_begin(overlay, completed, &frame);
		lc_overlay_end(overlay, 0, false);
		if (done()) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	return false;
}

void log_line(void*, lc_log_level, const char* line) {
	if (std::getenv("LC_TEST_LOG")) {
		std::fprintf(stderr, "log: %s\n", line);
	}
}

// A session whose overlay has the shell's surfaces, shown, ready for the trade
struct Shown {
	FakeLauncher launcher;
	lc_session*  session = nullptr;
	lc_overlay*  overlay = nullptr;
	lc_frame     frame {};

	explicit Shown(FakeShell& shell) {
		lc_callbacks cb {};
		cb.log = log_line;
		lc_connect_info info {};
		info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
		session           = lc_connect(&info, &cb, nullptr, 0);
		overlay           = lc_overlay_get(session);
		if (overlay == nullptr) {
			return;
		}
		launcher.send(R"({"type":"overlay_surface","buffers":[11,12,13],"header":10,"service":")" + shell.name() +
		              R"(","width":64,"height":64})");
		launcher.send(R"({"type":"overlay_visible","visible":true})");
		present_until(overlay, 0, frame, [&]() { return frame.buffer_count == 3 && lc_overlay_wants_presents(overlay); });
		lc_overlay_surfaces_ready(overlay, frame.generation, true);
	}
	~Shown() { lc_close(session); }
};

void overlay() {
	FakeShell    shell;
	FakeLauncher launcher;
	lc_callbacks cb {};
	cb.log = log_line;
	lc_connect_info info {};
	info.name         = "Test Emu";
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	CHECK(s != nullptr);
	if (s == nullptr) {
		return;
	}
	const auto hello = launcher.hello();
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 2);

	lc_overlay* overlay = lc_overlay_get(s);
	CHECK(overlay != nullptr);
	if (overlay == nullptr) {
		lc_close(s);
		return;
	}

	// view waits for ready, and goes out once per size
	lc_overlay_view(overlay, 1280, 720);
	lc_ready(s, 0, 0);
	lc_overlay_view(overlay, 1280, 720);
	CHECK(launcher.wait_for("view").has_value());

	// Nothing yet
	lc_frame frame {};
	lc_overlay_begin(overlay, 0, &frame);
	CHECK(frame.buffer_count == 0 && frame.draw == -1 && !frame.capture);

	// The menu opens: the next present copies its frame
	launcher.send(R"({"type":"overlay_opened"})");
	CHECK(present_until(overlay, 0, frame, [&]() { return frame.capture; }));
	CHECK(lc_overlay_wants_presents(overlay));
	lc_capture capture {};
	capture.fd     = make_memfd(64 * 64 * 4);
	capture.stride = 256;
	capture.width  = 64;
	capture.height = 64;
	lc_overlay_capture(overlay, &capture, 5);

	// The shell's surfaces
	launcher.send(R"({"type":"overlay_surface","buffers":[11,12,13],"header":10,"service":")" + shell.name() +
	              R"(","width":64,"height":64})");
	CHECK(present_until(overlay, 4, frame, [&]() { return frame.buffer_count == 3; }));
	CHECK(frame.generation == 1 && frame.width == 64 && frame.height == 64);
	CHECK(lc_surface_id(frame.buffers[0]) == 11 && lc_surface_id(frame.buffers[2]) == 13);
	CHECK(lc_surface_stride(frame.buffers[1]) == 256 && lc_surface_fd(frame.buffers[1]) >= 0);
	CHECK(shell.gets == 1);

	// Not before the surfaces are in use, nor before the GPU finished the copy (serial 5)
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECK(launcher.count("capture_surface") == 0);
	lc_overlay_surfaces_ready(overlay, frame.generation, true);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECK(launcher.count("capture_surface") == 0);
	lc_overlay_begin(overlay, 5, &frame);
	lc_overlay_end(overlay, 6, false);
	const auto id = shell.wait_capture();
	CHECK(id.has_value() && shell.capture_had_fd());
	const auto named = launcher.wait_for("capture_surface");
	CHECK(named && id && named->find("id")->as_uint() == *id);
	CHECK(named && named->find("width")->as_uint() == 64);

	// Shown: nothing to draw until the shell publishes
	launcher.send(R"({"type":"overlay_visible","visible":true})");
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	lc_overlay_begin(overlay, 5, &frame);
	CHECK(frame.draw == -1);
	lc_overlay_end(overlay, 6, false);

	// The shell publishes buffer 0, taking 1 from the word; the game takes 0 and gives back 2
	CHECK(shell.publish(0) == 1);
	lc_overlay_begin(overlay, 5, &frame);
	CHECK(frame.draw == 0 && frame.fresh);
	CHECK(shell.word() == 2);
	lc_overlay_end(overlay, 6, true);

	// Drawn again, not fresh
	lc_overlay_begin(overlay, 5, &frame);
	CHECK(frame.draw == 0 && !frame.fresh);
	lc_overlay_end(overlay, 7, true);

	// The shell publishes 1, but buffer 0 goes back only once serial 7 has finished
	CHECK(shell.publish(1) == 2);
	lc_overlay_begin(overlay, 6, &frame);
	CHECK(frame.draw == 0 && !frame.fresh && frame.wait == 7);
	CHECK((shell.word() & NEWER) != 0);
	// Begun again once 7 has finished, the same present takes it
	lc_overlay_begin(overlay, 7, &frame);
	CHECK(frame.draw == 1 && frame.fresh && frame.wait == 0);
	CHECK(shell.word() == 0);
	lc_overlay_end(overlay, 8, true);

	// A buffer index out of range breaks the exchange: nothing more is drawn from this set
	shell.publish(7);
	lc_overlay_begin(overlay, 100, &frame);
	CHECK(frame.draw == -1 && frame.buffer_count == 0);
	lc_overlay_end(overlay, 101, false);

	// Closing: hidden, then the surfaces go
	launcher.send(R"({"type":"overlay_visible","visible":false})");
	launcher.send(R"({"type":"overlay_closed"})");
	launcher.send(R"({"type":"overlay_surface","buffers":[],"width":0,"height":0})");
	CHECK(present_until(overlay, 100, frame, [&]() { return frame.generation == 2; }));
	CHECK(frame.buffer_count == 0 && frame.draw == -1);

	lc_close(s);
}

// A capture for a menu that closed before it was handed over is dropped
void closed_before_capture() {
	FakeLauncher launcher;
	lc_callbacks cb {};
	cb.log = log_line;
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	lc_overlay* o     = lc_overlay_get(s);
	CHECK(o != nullptr);
	if (o == nullptr) {
		lc_close(s);
		return;
	}
	lc_frame frame {};
	launcher.send(R"({"type":"overlay_opened"})");
	CHECK(present_until(o, 0, frame, [&]() { return frame.capture; }));
	launcher.send(R"({"type":"overlay_closed"})");
	CHECK(present_until(o, 0, frame, [&]() { return !lc_overlay_wants_presents(o); }));
	lc_capture capture {};
	capture.fd = make_memfd(16);
	lc_overlay_capture(o, &capture, 1);
	CHECK(!lc_overlay_wants_presents(o));
	lc_close(s);
}

// A renderer with two frames in flight draws every present: `completed` is always two behind, so
// the buffer to give back is always still being read, and the trade goes through by waiting
void frames_in_flight() {
	FakeShell shell;
	Shown     shown(shell);
	CHECK(shown.overlay != nullptr);
	if (shown.overlay == nullptr) {
		return;
	}
	uint32_t shell_has = 0; // The shell's own buffer
	int      taken     = 0;
	for (uint64_t serial = 1; serial <= 40; serial++) {
		// The shell publishes a new frame of the menu whenever the game took the last one
		if ((shell.word() & NEWER) == 0) {
			shell_has = shell.publish(shell_has);
		}
		lc_frame& f = shown.frame;
		lc_overlay_begin(shown.overlay, serial > 2 ? serial - 2 : 0, &f);
		if (f.wait != 0) {
			CHECK(f.wait < serial);
			lc_overlay_begin(shown.overlay, f.wait, &f);
			CHECK(f.wait == 0);
		}
		taken += f.fresh ? 1 : 0;
		CHECK(f.draw >= 0 && f.draw < 3 && static_cast<uint32_t>(f.draw) != shell_has);
		lc_overlay_end(shown.overlay, serial, true);
	}
	// Every present took the shell's newest
	CHECK(taken == 40);
}

// `fresh` holds until a present has drawn the buffer
void fresh_until_drawn() {
	FakeShell shell;
	Shown     shown(shell);
	CHECK(shown.overlay != nullptr);
	if (shown.overlay == nullptr) {
		return;
	}
	lc_frame& f = shown.frame;
	shell.publish(0);
	lc_overlay_begin(shown.overlay, 0, &f);
	CHECK(f.draw == 0 && f.fresh);
	lc_overlay_end(shown.overlay, 1, false);
	lc_overlay_begin(shown.overlay, 1, &f);
	CHECK(f.draw == 0 && f.fresh);
	lc_overlay_end(shown.overlay, 2, true);
	lc_overlay_begin(shown.overlay, 2, &f);
	CHECK(f.draw == 0 && !f.fresh);
	lc_overlay_end(shown.overlay, 3, true);
}

// Surfaces the emulator couldn't import: no capture goes out, and presents aren't wanted for it
void import_failed() {
	FakeShell    shell;
	FakeLauncher launcher;
	lc_callbacks cb {};
	cb.log = log_line;
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	lc_overlay* o     = lc_overlay_get(s);
	CHECK(o != nullptr);
	if (o == nullptr) {
		lc_close(s);
		return;
	}
	lc_frame frame {};
	launcher.send(R"({"type":"overlay_opened"})");
	launcher.send(R"({"type":"overlay_surface","buffers":[11,12,13],"header":10,"service":")" + shell.name() +
	              R"(","width":64,"height":64})");
	CHECK(present_until(o, 0, frame, [&]() { return frame.capture && frame.buffer_count == 3; }));
	lc_capture capture {};
	capture.fd     = make_memfd(64 * 64 * 4);
	capture.width  = 64;
	capture.height = 64;
	lc_overlay_capture(o, &capture, 1);
	lc_overlay_surfaces_ready(o, frame.generation, false);
	// Only the few presents owed for the change, none for the capture
	int presents = 0;
	CHECK(present_until(o, 5, frame, [&]() { return ++presents > 3 || !lc_overlay_wants_presents(o); }) && presents <= 3);
	CHECK(frame.buffer_count == 0 && frame.draw == -1);
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	CHECK(launcher.count("capture_surface") == 0);
	lc_close(s);
}

// A header that isn't sealed against shrinking is refused
void unsealed_header() {
	FakeShell shell({}, 256, false);
	Shown     shown(shell);
	CHECK(shown.overlay != nullptr && shown.frame.buffer_count == 0);
}

} // namespace

int main() {
	if (!lc_overlay_supported()) {
		std::printf("overlay: skipped (no backend on this platform)\n");
		return 0;
	}
	overlay();
	closed_before_capture();
	frames_in_flight();
	fresh_until_drawn();
	import_failed();
	unsealed_header();
	return finish("overlay");
}
