// Everything the library sends and reads, against the launcher's own examples
// (docs/plugin-api/examples, written from crates/proto's types), so the two can't drift apart.
#include "check.h"
#include "json.h"
#include "protocol.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using lc::json::Value;
namespace protocol = lc::protocol;

static std::string g_dir = LC_EXAMPLES_DIR;

static const char* type_name(const Value& v) {
	if (v.is_bool()) {
		return "a bool";
	}
	if (v.is_number()) {
		return "a number";
	}
	if (v.is_string()) {
		return "a string";
	}
	if (v.is_array()) {
		return "an array";
	}
	if (v.is_object()) {
		return "an object";
	}
	return "null";
}

// examples/<side>/<type>.json, which has to be an object of that type
static std::optional<Value> load(const std::string& side, const std::string& type) {
	const std::string name = side + "/" + type + ".json";
	std::ifstream     file(g_dir + "/" + name, std::ios::binary);
	if (!file) {
		FAIL("%s: missing", name.c_str());
		return std::nullopt;
	}
	std::stringstream text;
	text << file.rdbuf();
	auto doc = lc::json::parse(text.str());
	if (!doc || !doc->is_object() || doc->text("type") != type) {
		FAIL("%s: not a JSON object of that type", name.c_str());
		return std::nullopt;
	}
	return doc;
}

// What the library builds may only have fields the example has, of the same JSON types, and has
// to have every field the example sets, except `optional` ones it may leave out. Objects, and the
// first object in arrays, are compared the same way.
static void compare(const std::string& where, const Value& built, const Value& example, const std::vector<std::string>& optional) {
	for (const auto& [key, mine]: built.members()) {
		const Value* theirs = example.find(key);
		if (theirs == nullptr) {
			FAIL("%s: sends '%s', which the launcher doesn't know", where.c_str(), key.c_str());
		} else if (!mine.is_null() && !theirs->is_null() && std::string(type_name(mine)) != type_name(*theirs)) {
			FAIL("%s.%s: sends %s, the launcher wants %s", where.c_str(), key.c_str(), type_name(mine), type_name(*theirs));
		} else if (mine.is_object() && theirs->is_object()) {
			compare(where + "." + key, mine, *theirs, optional);
		} else if (mine.is_array() && theirs->is_array() && !mine.items().empty() && !theirs->items().empty() &&
		           mine.items().front().is_object() && theirs->items().front().is_object()) {
			compare(where + "." + key + "[0]", mine.items().front(), theirs->items().front(), optional);
		}
	}
	for (const auto& [key, value]: example.members()) {
		if (!value.is_null() && built.find(key) == nullptr && std::find(optional.begin(), optional.end(), key) == optional.end()) {
			FAIL("%s: doesn't send '%s', which the launcher's example has", where.c_str(), key.c_str());
		}
	}
}

static void check_sent(const protocol::Message& msg, const std::vector<std::string>& optional = {}) {
	if (const auto example = load("plugin", msg.type)) {
		const auto built = lc::json::parse(msg.json);
		CHECK(built.has_value());
		if (built) {
			compare(msg.type, *built, *example, optional);
		}
	}
}

static std::optional<protocol::HostMessage> host(const std::string& type) {
	if (const auto doc = load("host", type)) {
		return protocol::parse_host(doc->dump());
	}
	return std::nullopt;
}

static void sent() {
	check_sent(protocol::booting());
	check_sent(protocol::ready(4242, 77));
	check_sent(protocol::paused(true));
	check_sent(protocol::failed("The PS3 firmware isn't installed.", "firmware_missing"));
	check_sent(protocol::activity("shaders", "Compiling shaders", 0.5f));
	check_sent(protocol::activity_done("shaders"));
	check_sent(protocol::presence("Boletaria: Gates of Boletaria"));
	check_sent(protocol::notify("State saved", "Slot 1", false));

	lc_achievement full {};
	full.id              = "first-steps";
	full.name            = "First Steps";
	full.description     = "Leave the starting village.";
	full.grade           = "bronze";
	full.points          = 10;
	full.rarity_permille = 782;
	full.rarity_of       = "players";
	full.unlocked        = true;
	full.unlocked_at     = 1700000000;
	check_sent(protocol::achievements(&full, 1));
	// Without points or rarity, which most emulators don't know
	lc_achievement bare {};
	bare.id   = "1";
	bare.name = "First";
	check_sent(protocol::achievements(&bare, 1), {"grade", "points", "rarity", "unlocked_at"});

	check_sent(protocol::unlocked("first-steps", 1700000000));
	check_sent(protocol::achievement_progress("mushrooms", 3, 10));

	const lc_option slot_options[] = {{"1", "Slot 1"}};
	lc_setting      slot {};
	slot.key           = "slot";
	slot.section       = "Game";
	slot.label         = "Slot";
	slot.help          = "Which save-state slot to use.";
	slot.kind          = LC_SETTING_CHOICE;
	slot.options       = slot_options;
	slot.option_count  = 1;
	slot.value         = "1";
	slot.default_value = "1";
	check_sent(protocol::overlay_items(&slot, 1));

	check_sent(protocol::view(3600, 2338));
	check_sent(protocol::capture_surface(20, 3600, 2338));

	const char*   languages[] = {"en", "ja"};
	lc_osk_params osk {};
	osk.kind           = LC_OSK_TEXT;
	osk.title          = "Name your save";
	osk.placeholder    = "Hero";
	osk.text           = "Ann";
	osk.cursor         = 3;
	osk.max_length     = 16;
	osk.languages      = languages;
	osk.language_count = 2;
	osk.has_anchor     = true;
	osk.x              = 0.5f;
	osk.y              = 0.95f;
	osk.align_x        = LC_OSK_ALIGN_CENTER;
	osk.align_y        = LC_OSK_ALIGN_END;
	check_sent(protocol::osk_open(1, osk));
	{
		const auto   built  = lc::json::parse(protocol::osk_open(1, osk).json);
		const Value* anchor = built ? built->find("anchor") : nullptr;
		CHECK(anchor && anchor->find("x")->as_uint() == 500 && anchor->find("y")->as_uint() == 950);
	}
	// A bare one leaves out what it doesn't set
	lc_osk_params empty {};
	check_sent(protocol::osk_open(2, empty), {"title", "placeholder", "max_length", "languages", "anchor"});
	check_sent(protocol::osk_text(1, 4, "Anné", 4));
	check_sent(protocol::osk_closed(1, LC_OSK_ACCEPTED));
	lc_input arrow {};
	arrow.kind = LC_INPUT_KEY;
	arrow.key  = "ArrowUp";
	arrow.down = true;
	arrow.mods = LC_MOD_SHIFT;
	check_sent(protocol::overlay_input(arrow));
	{
		// The rest of the kinds, as the launcher reads them
		const auto input = [](const lc_input& in) {
			const auto built = lc::json::parse(protocol::overlay_input(in).json);
			return built ? std::optional<Value>(*built->find("input")) : std::nullopt;
		};
		lc_input text {};
		text.kind = LC_INPUT_TEXT;
		text.text = "é";
		CHECK(input(text)->text("kind") == "text" && input(text)->text("text") == "é");
		lc_input wheel {};
		wheel.kind = LC_INPUT_WHEEL;
		wheel.y    = -1.5f;
		CHECK(input(wheel)->find("y")->as_int() == -150 && !input(wheel)->find("pixels")->as_bool());
		lc_input button {};
		button.kind   = LC_INPUT_BUTTON;
		button.button = 1;
		button.down   = true;
		CHECK(input(button)->find("button")->as_uint() == 1 && input(button)->find("down")->as_bool());
		// Nothing to say
		lc_input nameless {};
		CHECK(protocol::overlay_input(nameless).type.empty());
	}

	// A plugin runner's
	lc_game portal {};
	portal.external_id = "620";
	portal.title       = "Portal 2";
	portal.platform    = "Pc";
	portal.installed   = true;
	portal.game_dir    = "/games/steamapps/common/Portal 2";
	portal.art.cover   = "/steam/appcache/librarycache/620/library_600x900.jpg";
	const char* warning = "No Web API key, so only installed games are listed.";
	check_sent(protocol::library(&portal, 1, &warning, 1), {"playtime_secs", "last_played"});
	check_sent(protocol::library_changed(&portal, 1), {"playtime_secs", "last_played"});

	const char* const args[] = {"--no-gui", "/games/BLUS30443/PS3_GAME/USRDIR/EBOOT.BIN"};
	const lc_kv       env[]  = {{"RPCS3_LOG", "1"}};
	lc_spawn          spawn {};
	spawn.exe       = "/emu/rpcs3";
	spawn.args      = args;
	spawn.arg_count = 2;
	spawn.env       = env;
	spawn.env_count = 1;
	spawn.cwd       = "/emu";
	check_sent(protocol::launch_spawn(3, spawn));
	check_sent(protocol::launch_error(3, "No such game"));
	check_sent(protocol::achievement_list(7, &full, 1));

	lc_setting key {};
	key.key         = "api_key";
	key.section     = "Account";
	key.label       = "Web API key";
	key.help        = "For owned games and achievements without the Steam client.";
	key.kind        = LC_SETTING_TEXT;
	key.placeholder = "Not set";
	check_sent(protocol::settings("Steam", &key, 1), {"secret", "set"});

	// A folder list, against the example's second row
	const lc_path_pick folders_pick {false, "Choose a folder of PS3 games", "~/Games", true};
	lc_setting         folders {};
	folders.key           = "game_dirs";
	folders.section       = "Library";
	folders.label         = "Game folders";
	folders.help          = "Folders of PS3 games, each in a folder of its own.";
	folders.kind          = LC_SETTING_PATH;
	folders.path          = &folders_pick;
	folders.value         = "/run/media/deck/SD/ps3\n/mnt/games/ps3";
	folders.default_value = "";
	if (const auto example = load("plugin", "settings")) {
		const auto built = lc::json::parse(protocol::settings("", &folders, 1).json);
		const auto& rows = example->find("items")->items();
		CHECK(built && rows.size() == 2);
		if (built && rows.size() == 2) {
			compare("settings.items[1]", built->find("items")->items().front(), rows[1], {"start"});
			const Value& row = built->find("items")->items().front();
			CHECK(row.find("kind")->find("Path")->find("multiple")->as_bool());
			CHECK(row.find("value")->find("List")->items().size() == 2);
			CHECK(row.find("default")->find("List")->items().empty());
		}
	}

	// Game settings: rows as settings' are, with `live` and the group of the page they go on
	const lc_option renderers[] = {{"Vulkan", "Vulkan"}, {"Metal", "Metal"}};
	lc_setting      renderer {};
	renderer.key           = "Video.Renderer";
	renderer.section       = "Graphics";
	renderer.label         = "Renderer";
	renderer.kind          = LC_SETTING_CHOICE;
	renderer.options       = renderers;
	renderer.option_count  = 2;
	renderer.value         = "Vulkan";
	renderer.default_value = "Vulkan";
	renderer.group         = "Emulators";
	lc_setting vsync {};
	vsync.key           = "Video.VSync";
	vsync.section       = "Graphics";
	vsync.label         = "VSync";
	vsync.kind          = LC_SETTING_TOGGLE;
	vsync.value         = "false";
	vsync.default_value = "false";
	vsync.group         = "Emulators";
	vsync.live          = true;
	const lc_setting game_rows[] = {renderer, vsync};
	check_sent(protocol::game_settings("RPCS3", game_rows, 2));
	const auto game = lc::json::parse(protocol::game_settings("RPCS3", game_rows, 2).json);
	CHECK(game && game->text("type") == "game_settings" && game->find("items")->items().size() == 2);
	if (game && game->find("items")->items().size() == 2) {
		const Value& first = game->find("items")->items()[0];
		const Value& second = game->find("items")->items()[1];
		CHECK(first.text("group") == "Emulators" && first.text("page") == "RPCS3" && !first.find("live")->as_bool());
		CHECK(second.find("live")->as_bool() && second.find("value")->find("Bool") != nullptr);
	}
}

static void values() {
	// Nulls where the protocol has them
	const auto ready = lc::json::parse(protocol::ready(1, 0).json);
	CHECK(ready && ready->find("window") != nullptr && ready->find("window")->is_null());
	const auto unlocked = lc::json::parse(protocol::unlocked("a", LC_UNLOCKED_NOW).json);
	CHECK(unlocked && unlocked->find("at")->is_null());
	const auto at_zero = lc::json::parse(protocol::unlocked("a", 0).json);
	CHECK(at_zero && at_zero->find("at")->is_unsigned() && at_zero->find("at")->as_uint() == 0);
	const auto none = lc::json::parse(protocol::activity("a", "b", -1).json);
	CHECK(none && none->find("progress")->is_null());
	const auto clear = lc::json::parse(protocol::presence(nullptr).json);
	CHECK(clear && clear->find("text")->is_null());

	// A runner's hello declares what it provides, and no session
	const auto runner_hello = lc::json::parse(protocol::hello("tok", 2, LC_CAP_LIBRARY | LC_CAP_LAUNCH | LC_CAP_ACHIEVEMENTS, "Emu 1").json);
	if (runner_hello) {
		std::vector<std::string> caps;
		for (const auto& cap: runner_hello->find("plugin")->find("capabilities")->items()) {
			caps.push_back(cap.as_string());
		}
		CHECK(caps == std::vector<std::string>({"library", "launch", "achievements"}));
	}
	const auto spawned = lc::json::parse(protocol::launch_spawn(1, lc_spawn {"/emu", nullptr, 0, nullptr, 0, nullptr}).json);
	CHECK(spawned && spawned->find("spawn")->find("cwd")->is_null() && spawned->find("error")->is_null());
	const auto refused = lc::json::parse(protocol::launch_error(1, "no").json);
	CHECK(refused && refused->find("spawn")->is_null() && refused->text("error") == "no");

	// The hello declares the overlay only when asked
	const auto hello = lc::json::parse(protocol::hello("tok", 2, LC_CAP_COMPOSITED_OVERLAY, "Emu 1").json);
	CHECK(hello && hello->find("plugin")->text("token") == "tok");
	CHECK(hello && hello->find("plugin")->find("capabilities")->items().size() == 2);
}

static void read() {
	// The values crates/proto/tests/examples.rs writes: a field it renamed reads as empty here
	if (const auto msg = host("accepted")) {
		CHECK(msg->api == LC_PLUGIN_API && msg->title == "Demon's Souls");
	}
	if (const auto msg = host("setting")) {
		CHECK(msg->key == "api_key" && msg->value == "0123456789ABCDEF");
	}
	// A folder list's value, one per line
	const auto folders = protocol::parse_host(R"({"type":"setting","key":"game_dirs","value":{"List":["/a","/b c"]}})");
	CHECK(folders.value == std::string("/a\n/b c"));
	if (const auto msg = host("invoke")) {
		CHECK(msg->key == "slot" && msg->value == "2");
	}
	if (const auto msg = host("configure")) {
		const protocol::Values want {{"Video.Renderer", "Vulkan"}, {"Video.Resolution Scale", "150"}};
		CHECK(msg->values == want);
	}
	if (const auto msg = host("overlay_surface")) {
		CHECK(msg->buffers == std::vector<uint32_t>({11, 12, 13}));
		CHECK(msg->header == 10 && !msg->service.empty() && msg->width == 1920 && msg->height == 1080 && !msg->lists);
		auto lists = *load("host", "overlay_surface");
		lists.set("lists", true);
		CHECK(protocol::parse_host(lists.dump()).lists);
	}
	if (const auto msg = host("overlay")) {
		CHECK(msg->pause && msg->input);
	}
	if (const auto msg = host("launch")) {
		CHECK(msg->launch == 3 && msg->game == 7 && msg->external_id == "620");
		CHECK(msg->game_dir == "/games/steamapps/common/Portal 2" && msg->args == std::vector<std::string>({"-novid"}));
	}
	if (const auto msg = host("list_achievements")) {
		CHECK(msg->game == 7 && msg->external_id == "620" && msg->game_dir == "/games/steamapps/common/Portal 2");
	}
	if (const auto msg = host("osk_available")) {
		CHECK(msg->available);
	}
	if (const auto msg = host("osk_edit")) {
		CHECK(msg->osk == 1 && msg->seq == 4 && msg->edit_ok && msg->edit_op == LC_OSK_INSERT && msg->edit_text == "\u00e9");
	}
	if (const auto msg = host("osk_finish")) {
		CHECK(msg->osk == 1 && msg->accepted);
	}
	// Every edit the launcher sends, and one from a newer one
	const auto edit = [](const char* json) {
		return protocol::parse_host(std::string(R"({"type":"osk_edit","osk":2,"seq":9,"edit":)") + json + "}");
	};
	CHECK(edit(R"({"op":"backspace"})").edit_op == LC_OSK_BACKSPACE);
	CHECK(edit(R"({"op":"delete"})").edit_op == LC_OSK_DELETE);
	const auto left = edit(R"({"op":"move","by":-3})");
	CHECK(left.edit_ok && left.edit_op == LC_OSK_MOVE && left.edit_by == -3);
	const auto replace = edit(R"({"op":"replace","text":"\u65e5\u672c","cursor":2})");
	CHECK(replace.edit_ok && replace.edit_op == LC_OSK_REPLACE && replace.edit_text == "\u65e5\u672c" && replace.edit_cursor == 2);
	CHECK(!edit(R"({"op":"select_all"})").edit_ok);
	CHECK(!edit(R"({"op":"move","by":"x"})").edit_ok);
	for (const char* type: {"quit", "scan"}) {
		if (const auto msg = host(type)) {
			CHECK(msg->type == type);
		}
	}
	// Older launchers spell the encoding "Json"
	CHECK(protocol::parse_host(R"({"encoding":"Json","server":"launcherd 0.1"})").encoding == "json");
	CHECK(!protocol::parse_host("not json").ok);
}

static void json() {
	const auto v = lc::json::parse(R"({"a":[1,-2,3.5,true,null,"x\u00e9\ud83d\ude00\n"],"b":{}})");
	CHECK(v.has_value());
	if (v) {
		CHECK(v->dump() == "{\"a\":[1,-2,3.5,true,null,\"x\xC3\xA9\xF0\x9F\x98\x80\\n\"],\"b\":{}}");
	}
	CHECK(!lc::json::parse("{\"a\":1,}"));
	CHECK(!lc::json::parse("[1] x"));
	CHECK(!lc::json::parse(std::string(100, '[') + std::string(100, ']')));
	// Bad UTF-8 is replaced, not passed on
	CHECK(Value(std::string("a\xFF" "b")).dump() == "\"a\xEF\xBF\xBD" "b\"");
	CHECK(Value(std::string("\x01")).dump() == "\"\\u0001\"");
}

int main(int argc, char** argv) {
	if (argc > 1) {
		g_dir = argv[1];
	}
	json();
	sent();
	values();
	read();
	return finish("protocol");
}
