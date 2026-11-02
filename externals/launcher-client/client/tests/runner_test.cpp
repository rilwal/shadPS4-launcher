// A plugin runner against a fake launcherd: the hello, what launcherd asks through the runner
// callbacks, and the answers going back.
#include "check.h"
#include "fake_launcher.h"
#include "launcher_overlay.h"
#include "launcher_runner.h"

#include <map>

namespace {

struct Seen {
	std::mutex                         mutex;
	std::condition_variable            cv;
	int                                scans = 0;
	std::vector<std::string>           launched;
	std::vector<std::string>           listed;
	std::map<std::string, std::string> settings;

	template <typename F>
	bool wait(F done) {
		std::unique_lock lock(mutex);
		return cv.wait_for(lock, std::chrono::seconds(2), done);
	}
};

void runner() {
	FakeLauncher launcher({
	    R"({"type":"accepted","api":2,"game":null,"title":""})",
	    R"({"type":"setting","key":"game_dirs","value":{"Text":"/roms/ps5:/more"}})",
	    R"({"type":"scan"})",
	    R"({"type":"list_achievements","game":7,"runner":"kyty","external_id":"PPSA21564","game_dir":"/roms/ps5/astro"})",
	    R"({"type":"launch","launch":3,"game":7,"external_id":"PPSA21564","game_dir":"/roms/ps5/astro","args":null,"session":"s"})",
	});
	Seen         seen;
	lc_callbacks cb {};
	cb.user    = &seen;
	cb.log     = [](void*, lc_log_level, const char*) {};
	cb.setting = [](void* user, const char* key, const char* value) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->settings[key] = value != nullptr ? value : "(null)";
		s->cv.notify_all();
	};
	lc_runner_callbacks rc {};
	rc.user = &seen;
	rc.scan = [](void* user) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->scans++;
		s->cv.notify_all();
	};
	rc.launch = [](void* user, const lc_launch_request* r) {
		auto* s = static_cast<Seen*>(user);
		// Answer from the callback, as a runner may
		const char* const args[] = {"--game", r->game_dir};
		lc_spawn          spawn {};
		spawn.exe       = "/emu/kyty_emulator";
		spawn.args      = args;
		spawn.arg_count = 2;
		lc_launch_spawn(lc_get(), r->launch, &spawn);
		std::lock_guard lock(s->mutex);
		s->launched.push_back(std::string(r->external_id) + "@" + r->game_dir + " args:" + std::to_string(r->arg_count));
		s->cv.notify_all();
	};
	rc.list_achievements = [](void* user, uint64_t game, const char* external_id, const char* game_dir) {
		auto*           s = static_cast<Seen*>(user);
		std::lock_guard lock(s->mutex);
		s->listed.push_back(std::to_string(game) + ":" + external_id + "@" + game_dir);
		s->cv.notify_all();
	};

	lc_connect_info info {};
	info.name         = "Kyty test";
	info.capabilities = LC_CAP_LIBRARY | LC_CAP_LAUNCH | LC_CAP_ACHIEVEMENTS | LC_CAP_SETTINGS;
	char        error[256] {};
	lc_session* s = lc_runner_connect(&info, &cb, &rc, error, sizeof(error));
	if (s == nullptr) {
		FAIL("lc_runner_connect: %s", error);
		return;
	}

	const auto hello = launcher.hello();
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 4);
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().front().as_string() == "library");

	CHECK(seen.wait([&]() { return seen.scans == 1 && seen.listed.size() == 1 && seen.launched.size() == 1 && seen.settings.size() == 1; }));
	{
		std::lock_guard lock(seen.mutex);
		CHECK(seen.settings["game_dirs"] == "/roms/ps5:/more");
		CHECK(!seen.listed.empty() && seen.listed[0] == "7:PPSA21564@/roms/ps5/astro");
		CHECK(!seen.launched.empty() && seen.launched[0] == "PPSA21564@/roms/ps5/astro args:0");
	}
	const auto spawn = launcher.wait_for("launch_result");
	CHECK(spawn && spawn->find("launch")->as_uint() == 3);
	CHECK(spawn && spawn->find("spawn")->text("exe") == "/emu/kyty_emulator");
	CHECK(spawn && spawn->find("spawn")->find("args")->items().size() == 2);

	lc_game astro {};
	astro.external_id = "PPSA21564";
	astro.title       = "ASTRO BOT";
	astro.platform    = "Ps5";
	astro.installed   = true;
	astro.game_dir    = "/roms/ps5/astro";
	astro.art.icon    = "/roms/ps5/astro/sce_sys/icon0.png";
	lc_library(s, &astro, 1, nullptr, 0);
	const auto library = launcher.wait_for("library");
	CHECK(library && library->find("games")->items().size() == 1);
	CHECK(library && library->find("games")->items()[0].text("platform") == "Ps5");
	CHECK(library && library->find("games")->items()[0].find("art")->text("icon") == "/roms/ps5/astro/sce_sys/icon0.png");
	CHECK(library && library->find("games")->items()[0].find("art")->find("hero")->is_null());

	lc_achievement cup {};
	cup.id   = "0";
	cup.name = "Astro Bot";
	lc_achievement_list(s, 7, &cup, 1);
	const auto list = launcher.wait_for("achievement_list");
	CHECK(list && list->find("game")->as_uint() == 7 && list->find("list")->items().size() == 1);

	lc_setting dirs {};
	dirs.key   = "game_dirs";
	dirs.label = "Game folders";
	dirs.kind  = LC_SETTING_TEXT;
	lc_settings(s, "Kyty", &dirs, 1);
	const auto settings = launcher.wait_for("settings");
	CHECK(settings && settings->find("items")->items()[0].text("page") == "Kyty");

	lc_close(s);
}

// Which kind of connection comes from the function, not the flags
void kinds() {
	lc_connect_info info {};
	char            error[256] {};
	{
		FakeLauncher        launcher;
		lc_runner_callbacks rc {};
		info.capabilities = LC_CAP_COMPOSITED_OVERLAY;
		CHECK(lc_runner_connect(&info, nullptr, &rc, error, sizeof(error)) == nullptr);
		CHECK(std::string(error).find("runner capability") != std::string::npos);
	}
	{
		// A runner's overlay capability is its game sessions': launcherd adds no hook to them
		FakeLauncher launcher;
		info.capabilities = LC_CAP_LIBRARY | LC_CAP_COMPOSITED_OVERLAY;
		lc_session* s     = lc_runner_connect(&info, nullptr, nullptr, error, sizeof(error));
		CHECK(s != nullptr);
		const auto hello = launcher.hello();
		CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 2);
		CHECK(hello && hello->find("plugin")->find("capabilities")->items()[0].as_string() == "composited_overlay");
		CHECK(lc_overlay_get(s) == nullptr);
		lc_close(s);
	}
	{
		FakeLauncher launcher;
		info.capabilities = LC_CAP_LIBRARY;
		lc_session* s     = lc_connect(&info, nullptr, error, sizeof(error));
		CHECK(s != nullptr);
		const auto hello = launcher.hello();
		CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 1);
		CHECK(hello && hello->find("plugin")->find("capabilities")->items()[0].as_string() == "session");
		lc_close(s);
	}
}

} // namespace

int main() {
	runner();
	kinds();
	return finish("runner");
}
