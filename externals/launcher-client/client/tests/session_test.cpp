// A whole session against a fake launcherd: the handshake, configure, what goes out, and what
// comes back through the callbacks.
#include "check.h"
#include "fake_launcher.h"
#include "launcher_client.h"

#include <atomic>
#include <map>

namespace {

struct Seen {
	std::mutex                         mutex;
	std::condition_variable            cv;
	std::atomic<int>                   quit {0};
	std::vector<bool>                  overlay;
	std::map<std::string, std::string> invoked;
	std::map<std::string, std::string> settings;
	std::vector<bool>                  osk_available;
	std::vector<std::string>           osk_edits; // "<osk>:<seq>:<op>:<text>:<by>:<cursor>"
	std::vector<std::string>           osk_finished;

	template <typename F>
	bool wait(F done) {
		std::unique_lock lock(mutex);
		return cv.wait_for(lock, std::chrono::seconds(2), done);
	}
};

lc_callbacks callbacks(Seen& seen) {
	lc_callbacks cb {};
	cb.user = &seen;
	cb.log  = [](void*, lc_log_level, const char*) {};
	cb.quit = [](void* user) {
		auto* s = static_cast<Seen*>(user);
		s->quit++;
		s->cv.notify_all();
	};
	cb.overlay = [](void* user, bool opened) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->overlay.push_back(opened);
		s->cv.notify_all();
	};
	cb.invoke = [](void* user, const char* key, const char* value) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->invoked[key] = value != nullptr ? value : "(null)";
		s->cv.notify_all();
	};
	cb.setting = [](void* user, const char* key, const char* value) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->settings[key] = value != nullptr ? value : "(null)";
		s->cv.notify_all();
	};
	cb.osk_available = [](void* user, bool available) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->osk_available.push_back(available);
		s->cv.notify_all();
	};
	cb.osk_edit = [](void* user, uint32_t osk, uint64_t seq, const lc_osk_edit* e) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->osk_edits.push_back(std::to_string(osk) + ":" + std::to_string(seq) + ":" + std::to_string(e->op) + ":" + e->text +
		                       ":" + std::to_string(e->by) + ":" + std::to_string(e->cursor));
		s->cv.notify_all();
	};
	cb.osk_finish = [](void* user, uint32_t osk, bool accepted) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->osk_finished.push_back(std::to_string(osk) + (accepted ? ":accepted" : ":cancelled"));
		s->cv.notify_all();
	};
	return cb;
}

void session() {
	FakeLauncher launcher({
	    R"({"type":"accepted","api":2,"game":7,"title":"Demon's Souls"})",
	    R"({"type":"configure","values":{"Video.Renderer":"Vulkan","Video.Scale":{"Text":"150"},"Audio.Off":null}})",
	});
	Seen          seen;
	const auto    cb = callbacks(seen);
	lc_connect_info info {};
	info.name = "Test Emu 1.0";
	char        error[256] {};
	lc_session* s = lc_connect(&info, &cb, error, sizeof(error));
	CHECK(s != nullptr);
	if (s == nullptr) {
		FAIL("lc_connect: %s", error);
		return;
	}
	CHECK(lc_get() == s);

	// The hello: the token, the API, and only "session" without the overlay
	const auto hello = launcher.hello();
	CHECK(hello && hello->find("plugin")->text("token") == "test-token");
	CHECK(hello && hello->find("plugin")->find("api")->as_uint() == LC_PLUGIN_API);
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 1);
	CHECK(hello && hello->text("client") == "Test Emu 1.0");

	const lc_kv* values = nullptr;
	size_t       count  = 0;
	CHECK(lc_wait_configure(s, 2000, &values, &count));
	CHECK(count == 3);
	if (count == 3) {
		CHECK(std::string(values[0].key) == "Video.Renderer" && std::string(values[0].value) == "Vulkan");
		CHECK(std::string(values[1].key) == "Video.Scale" && std::string(values[1].value) == "150");
		CHECK(std::string(values[2].key) == "Audio.Off" && values[2].value == nullptr);
	}
	CHECK(std::string(lc_title(s)) == "Demon's Souls");
	CHECK(lc_api(s) == 2);

	lc_booting(s);
	lc_activity(s, "shaders", "Compiling shaders", 0.1f);
	// Coalesced or throttled: the same label within 100 ms
	lc_activity(s, "shaders", "Compiling shaders", 0.2f);
	lc_activity(s, "shaders", "Compiling shaders", 0.3f);
	lc_activity_done(s, "shaders");
	// Not started, so not sent
	lc_activity_done(s, "never");
	lc_ready(s, 0, 0);
	lc_paused(s, true);
	lc_presence(s, "Boletaria");
	lc_notify(s, "State saved", "Slot 1", false);
	lc_achievement a {};
	a.id   = "1";
	a.name = "First";
	lc_achievements(s, &a, 1);
	lc_unlocked(s, "1", LC_UNLOCKED_NOW);
	lc_achievement_progress(s, "2", 3, 10);
	lc_setting item {};
	item.key  = "save";
	item.kind = LC_SETTING_ACTION;
	lc_overlay_items(s, &item, 1);
	CHECK(lc_flush(s, 2000));

	CHECK(launcher.wait_for("overlay_items").has_value());
	CHECK(launcher.count("activity") == 1);
	CHECK(launcher.count("activity_done") == 1);
	const auto ready = launcher.wait_for("ready");
	CHECK(ready && ready->find("pid")->as_uint() == static_cast<uint64_t>(::getpid()));
	CHECK(ready && ready->find("window")->is_null());
	const std::vector<std::string> want {"booting",      "activity",     "activity_done", "ready",
	                                     "paused",       "presence",     "notify",        "achievements",
	                                     "unlocked",     "achievement_progress", "overlay_items"};
	CHECK(launcher.types() == want);

	// What the launcher asks for
	launcher.send(R"({"type":"invoke","key":"save","value":null,"game":null})");
	launcher.send(R"({"type":"invoke","key":"slot","value":{"Text":"2"}})");
	launcher.send(R"({"type":"setting","key":"perf-overlay","value":{"Bool":true}})");
	launcher.send(R"({"type":"overlay_opened"})");
	launcher.send(R"({"type":"something_newer","x":1})");
	launcher.send(R"({"type":"overlay_closed"})");
	launcher.send(R"({"type":"quit"})");
	CHECK(seen.wait([&]() { return seen.quit.load() == 1; }));
	{
		std::lock_guard lock(seen.mutex);
		CHECK(seen.invoked["save"] == "(null)");
		CHECK(seen.invoked["slot"] == "2");
		CHECK(seen.settings["perf-overlay"] == "true");
		CHECK(seen.overlay == std::vector<bool>({true, false}));
	}

	// failed goes out once, and is written before it returns
	lc_failed(s, "It broke.", "fatal_error");
	lc_failed(s, "Again.", nullptr);
	CHECK(launcher.wait_for("failed").has_value());
	CHECK(launcher.count("failed") == 1);

	lc_close(s);
	CHECK(lc_get() == nullptr);
}

// The on-screen keyboard: only while the launcher offers it, keys only for the one open, and
// given back when the offer ends
void osk() {
	FakeLauncher launcher({R"({"type":"accepted","api":4,"game":7,"title":"Demon's Souls"})"});
	Seen          seen;
	const auto    cb = callbacks(seen);
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY | LC_CAP_OSK;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	CHECK(s != nullptr);
	if (s == nullptr) {
		return;
	}
	const auto hello = launcher.hello();
	const auto caps  = hello ? hello->find("plugin")->find("capabilities")->dump() : std::string {};
	CHECK(caps == R"(["session","composited_overlay","osk"])");

	lc_osk_params params {};
	params.kind       = LC_OSK_LATIN;
	params.text       = "Ann";
	params.cursor     = 3;
	params.max_length = 16;
	// Not offered yet: the emulator shows its own
	CHECK(!lc_osk_available(s));
	CHECK(lc_osk_open(s, &params) == 0);

	launcher.send(R"({"type":"osk_available","available":true})");
	CHECK(seen.wait([&]() { return seen.osk_available == std::vector<bool>({true}); }));
	CHECK(lc_osk_available(s));
	const uint32_t first = lc_osk_open(s, &params);
	CHECK(first != 0);
	const auto open = launcher.wait_for("osk_open");
	CHECK(open && open->find("osk")->as_uint() == first && open->text("kind") == "latin" && open->text("text") == "Ann");
	CHECK(open && open->find("max_length")->as_uint() == 16 && open->find("title") == nullptr);

	// Keys for it, and for none that's open
	const auto key = [&](uint32_t osk, uint64_t seq, const char* edit) {
		launcher.send(std::string(R"({"type":"osk_edit","osk":)") + std::to_string(osk) + R"(,"seq":)" + std::to_string(seq) +
		              R"(,"edit":)" + edit + "}");
	};
	key(first, 1, R"({"op":"insert","text":"e"})");
	key(first + 100, 2, R"({"op":"backspace"})");
	key(first, 3, R"({"op":"move","by":-2})");
	key(first, 4, R"({"op":"replace","text":"x","cursor":1})");
	key(first, 5, R"({"op":"undo"})");
	CHECK(seen.wait([&]() { return seen.osk_edits.size() == 3; }));
	{
		std::lock_guard lock(seen.mutex);
		const auto      f = std::to_string(first);
		CHECK(seen.osk_edits == std::vector<std::string>({f + ":1:0:e:0:0", f + ":3:3::-2:0", f + ":4:4:x:0:1"}));
	}
	lc_osk_text(s, first, 3, "Anne", 2);
	lc_osk_text(s, first + 100, 3, "nope", 0);
	const auto text = launcher.wait_for("osk_text");
	CHECK(text && text->text("text") == "Anne" && text->find("seq")->as_uint() == 3 && text->find("cursor")->as_uint() == 2);

	// Another opens over it: the first closes first
	const uint32_t second = lc_osk_open(s, &params);
	CHECK(second != 0 && second != first);
	const auto aborted = launcher.wait_for("osk_closed");
	CHECK(aborted && aborted->find("osk")->as_uint() == first && aborted->text("result") == "aborted");
	CHECK(launcher.wait_for("osk_open", 1).has_value());
	launcher.send(R"({"type":"osk_finish","osk":)" + std::to_string(first) + R"(,"accepted":true})");
	launcher.send(R"({"type":"osk_finish","osk":)" + std::to_string(second) + R"(,"accepted":false})");
	CHECK(seen.wait([&]() { return seen.osk_finished.size() == 1; }));
	{
		std::lock_guard lock(seen.mutex);
		CHECK(seen.osk_finished == std::vector<std::string>({std::to_string(second) + ":cancelled"}));
	}
	lc_osk_close(s, second, LC_OSK_CANCELLED);
	lc_osk_close(s, second, LC_OSK_CANCELLED);
	CHECK(launcher.wait_for("osk_closed", 1).has_value());
	CHECK(lc_flush(s, 2000));
	CHECK(launcher.count("osk_closed") == 2);
	CHECK(launcher.count("osk_text") == 1);

	// The offer ends with one open: it's the emulator's again, launcherd is told it closed (in case
	// the open crossed the offer ending and arrived after a new one), and nothing more goes out for it
	const uint32_t third = lc_osk_open(s, &params);
	CHECK(launcher.wait_for("osk_open", 2).has_value());
	launcher.send(R"({"type":"osk_available","available":false})");
	CHECK(seen.wait([&]() { return seen.osk_available == std::vector<bool>({true, false}); }));
	const auto given_back = launcher.wait_for("osk_closed", 2);
	CHECK(given_back && given_back->find("osk")->as_uint() == third && given_back->text("result") == "aborted");
	lc_osk_text(s, third, 0, "late", 4);
	lc_osk_close(s, third, LC_OSK_ACCEPTED);
	CHECK(lc_osk_open(s, &params) == 0);
	CHECK(lc_flush(s, 2000));
	CHECK(launcher.count("osk_text") == 1 && launcher.count("osk_closed") == 3 && launcher.count("osk_open") == 3);

	// Offered again, then the launcher goes away: not offered any more
	launcher.send(R"({"type":"osk_available","available":true})");
	CHECK(seen.wait([&]() { return seen.osk_available.size() == 3; }));
	launcher.disconnect();
	CHECK(seen.wait([&]() { return seen.osk_available.size() == 4; }));
	CHECK(!lc_osk_available(s));
	lc_close(s);
}

// Closing the session doesn't call back that keyboards aren't offered, as it doesn't call closed
void osk_close_quietly() {
	FakeLauncher launcher({R"({"type":"accepted","api":4,"game":7,"title":"Demon's Souls"})"});
	Seen         seen;
	const auto   cb = callbacks(seen);
	lc_connect_info info {};
	info.capabilities = LC_CAP_COMPOSITED_OVERLAY | LC_CAP_OSK;
	lc_session* s     = lc_connect(&info, &cb, nullptr, 0);
	CHECK(s != nullptr);
	if (s == nullptr) {
		return;
	}
	launcher.send(R"({"type":"osk_available","available":true})");
	CHECK(seen.wait([&]() { return seen.osk_available.size() == 1; }));
	lc_close(s);
	std::lock_guard lock(seen.mutex);
	CHECK(seen.osk_available == std::vector<bool>({true}));
}

void no_configure() {
	FakeLauncher launcher;
	Seen         seen;
	const auto   cb = callbacks(seen);
	lc_session*  s  = lc_connect(nullptr, &cb, nullptr, 0);
	CHECK(s != nullptr);
	const auto start = std::chrono::steady_clock::now();
	CHECK(!lc_wait_configure(s, 200, nullptr, nullptr));
	CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
	lc_close(s);
}

// A launcherd that stops reading doesn't keep close from returning
void stuck_launcher() {
	FakeLauncher launcher;
	lc_session*  s = lc_connect(nullptr, nullptr, nullptr, 0);
	CHECK(s != nullptr);
	CHECK(launcher.hello().has_value());
	launcher.stop_reading();
	// Far more than the socket buffers hold
	const std::string text(16 * 1024, 'x');
	for (int i = 0; i < 400; i++) {
		lc_notify(s, text.c_str(), nullptr, false);
	}
	const auto start = std::chrono::steady_clock::now();
	lc_close(s);
	CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(3));
}

void outside_launcher() {
	::unsetenv("LAUNCHER_ENDPOINT");
	::unsetenv("LAUNCHER_SESSION");
	CHECK(!lc_env_present());
	char error[128] {};
	CHECK(lc_connect(nullptr, nullptr, error, sizeof(error)) == nullptr);
	CHECK(error[0] != '\0');
	// Calls on no session do nothing
	lc_booting(nullptr);
	lc_ready(nullptr, 0, 0);
	CHECK(lc_flush(nullptr, 0));
	CHECK(!lc_wait_configure(nullptr, 0, nullptr, nullptr));
	lc_close(nullptr);
}

} // namespace

int main() {
	session();
	osk();
	osk_close_quietly();
	no_configure();
	stuck_launcher();
	outside_launcher();
	return finish("session");
}
