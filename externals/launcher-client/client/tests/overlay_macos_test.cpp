// The composited overlay on macOS against a fake launcherd and a fake shell bootstrap service:
// IOSurfaces come as Mach ports and are checked by ID, the header's trade runs under its lock,
// and the capture goes back as a port before capture_surface names its IOSurface ID.
#include "check.h"
#include "fake_launcher.h"
#include "fake_shell_macos.h"
#include "launcher_overlay.h"

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

void overlay() {
	CHECK(lc_overlay_supported());
	FakeShell    shell;
	FakeLauncher launcher;
	lc_callbacks cb {};
	cb.log = log_line;
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	CHECK(s != nullptr);
	const auto hello = launcher.hello();
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 2);
	lc_overlay* overlay = lc_overlay_get(s);
	CHECK(overlay != nullptr);
	if (overlay == nullptr) {
		lc_close(s);
		return;
	}

	// The menu opens: the next present copies its frame, which the library retains
	launcher.send(R"({"type":"overlay_opened"})");
	lc_frame frame {};
	CHECK(present_until(overlay, 0, frame, [&]() { return frame.capture; }));
	IOSurfaceRef copied = make_surface(64, 64);
	const uint32_t copied_id = IOSurfaceGetID(copied);
	lc_capture   capture {};
	capture.fd        = -1;
	capture.iosurface = copied;
	capture.width     = 64;
	capture.height    = 64;
	lc_overlay_capture(overlay, &capture, 5);
	CFRelease(copied);

	// The shell's surfaces, by their IOSurface IDs
	launcher.send(shell.offer());
	CHECK(present_until(overlay, 4, frame, [&]() { return frame.buffer_count == 3; }));
	CHECK(frame.width == 64 && frame.height == 64 && shell.gets == 1);
	for (uint32_t i = 0; i < 3 && i < frame.buffer_count; i++) {
		auto* surface = static_cast<IOSurfaceRef>(lc_surface_iosurface(frame.buffers[i]));
		CHECK(lc_surface_id(frame.buffers[i]) == shell.buffer_id(i));
		CHECK(surface != nullptr && IOSurfaceGetID(surface) == shell.buffer_id(i));
		CHECK(lc_surface_stride(frame.buffers[i]) == IOSurfaceGetBytesPerRow(shell.buffer(i)));
		CHECK(lc_surface_fd(frame.buffers[i]) == -1);
	}

	// Handed back once the surfaces are in use and the copy (serial 5) has finished
	lc_overlay_surfaces_ready(overlay, frame.generation, true);
	lc_overlay_begin(overlay, 5, &frame);
	lc_overlay_end(overlay, 6, false);
	const auto id = shell.wait_capture();
	CHECK(id.has_value() && *id == copied_id);
	const auto named = launcher.wait_for("capture_surface");
	CHECK(named && named->find("id")->as_uint() == copied_id);

	// Shown: the shell publishes buffer 0, taking 1 from the word; the game takes 0 and gives back 2
	launcher.send(R"({"type":"overlay_visible","visible":true})");
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	lc_overlay_begin(overlay, 5, &frame);
	CHECK(frame.draw == -1);
	lc_overlay_end(overlay, 6, false);
	CHECK(shell.publish(0) == 1);
	lc_overlay_begin(overlay, 5, &frame);
	CHECK(frame.draw == 0 && frame.fresh);
	CHECK(shell.word() == 2);
	lc_overlay_end(overlay, 6, true);

	// Surfaces other than the ones named are refused: a set of none, so nothing to draw
	launcher.send(R"({"type":"overlay_surface","buffers":[1,2,3],"header":4,"service":")" + shell.name() +
	              R"(","width":64,"height":64})");
	CHECK(present_until(overlay, 6, frame, [&]() { return frame.generation == 2; }));
	CHECK(shell.gets == 2 && frame.buffer_count == 0 && frame.draw == -1);

	// So are buffers of another size than overlay_surface says
	launcher.send(shell.offer(32));
	CHECK(present_until(overlay, 7, frame, [&]() { return frame.generation == 3; }));
	CHECK(shell.gets == 3 && frame.buffer_count == 0);

	// And the right ones are taken again
	launcher.send(shell.offer());
	CHECK(present_until(overlay, 7, frame, [&]() { return frame.generation == 4; }));
	CHECK(shell.gets == 4 && frame.buffer_count == 3);

	lc_close(s);
}

// A service that isn't there: no surfaces, and the session carries on
void no_service() {
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
	launcher.send(R"({"type":"overlay_surface","buffers":[1,2,3],"header":4,"service":"lc-test.nobody","width":64,"height":64})");
	lc_frame frame {};
	CHECK(present_until(o, 0, frame, [&]() { return frame.generation == 1; }));
	CHECK(frame.buffer_count == 0);
	lc_close(s);
}

} // namespace

int main() {
	overlay();
	no_service();
	return finish("overlay_macos");
}
