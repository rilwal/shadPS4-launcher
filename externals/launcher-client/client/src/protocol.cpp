#include "protocol.h"

#include "json.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace lc::protocol {

using json::Value;

static std::string str(const char* s) {
	return s != nullptr ? std::string(s) : std::string {};
}

static Message make(const char* type, Value obj) {
	Message msg;
	msg.type = type;
	msg.json = obj.dump();
	return msg;
}

static Value typed(const char* type) {
	Value obj = Value::object();
	obj.set("type", type);
	return obj;
}

Message hello(const std::string& token, int api, uint32_t capabilities, const std::string& name) {
	Value caps = Value::array();
	if ((capabilities & LC_CAP_RUNNER) == 0) {
		caps.push("session");
	}
	const std::pair<uint32_t, const char*> names[] = {
	    {LC_CAP_COMPOSITED_OVERLAY, "composited_overlay"},
	    {LC_CAP_LIBRARY, "library"},
	    {LC_CAP_LAUNCH, "launch"},
	    {LC_CAP_ACHIEVEMENTS, "achievements"},
	    {LC_CAP_SETTINGS, "settings"},
	    {LC_CAP_OSK, "osk"},
	    {LC_CAP_DRAW_LISTS, "draw_lists"},
	};
	for (const auto& [flag, cap]: names) {
		if ((capabilities & flag) != 0) {
			caps.push(cap);
		}
	}
	Value plugin = Value::object();
	plugin.set("token", token);
	plugin.set("api", api);
	plugin.set("capabilities", std::move(caps));
	plugin.set("name", name);

	Value encodings = Value::array();
	encodings.push("json");

	Value obj = Value::object();
	obj.set("schema_hash", 0);
	obj.set("encodings", std::move(encodings));
	obj.set("client", name);
	obj.set("plugin", std::move(plugin));
	return make("hello", std::move(obj));
}

Message booting() {
	return make("booting", typed("booting"));
}

Message ready(uint32_t pid, uint64_t window) {
	Value obj = typed("ready");
	obj.set("pid", pid);
	obj.set("window", window != 0 ? Value(window) : Value());
	return make("ready", std::move(obj));
}

Message failed(const char* reason, const char* code) {
	Value obj = typed("failed");
	obj.set("reason", str(reason));
	if (code != nullptr && *code != '\0') {
		obj.set("code", code);
	}
	return make("failed", std::move(obj));
}

Message paused(bool paused) {
	Value obj = typed("paused");
	obj.set("paused", paused);
	return make("paused", std::move(obj));
}

Message activity(const char* id, const char* label, float progress) {
	Value obj = typed("activity");
	obj.set("id", str(id));
	obj.set("label", str(label));
	obj.set("progress", progress >= 0 ? Value(static_cast<double>(std::min(progress, 1.0f))) : Value());
	Message msg     = make("activity", std::move(obj));
	msg.activity_id = str(id);
	return msg;
}

Message activity_done(const char* id) {
	Value obj = typed("activity_done");
	obj.set("id", str(id));
	Message msg     = make("activity_done", std::move(obj));
	msg.activity_id = str(id);
	return msg;
}

Message presence(const char* text) {
	Value obj = typed("presence");
	obj.set("text", Value::text_or_null(text));
	return make("presence", std::move(obj));
}

Message notify(const char* title, const char* body, bool error) {
	Value obj = typed("notify");
	obj.set("title", str(title));
	obj.set("body", str(body));
	obj.set("level", error ? "error" : "info");
	return make("notify", std::move(obj));
}

static Value achievement(const lc_achievement& a) {
	Value obj = Value::object();
	obj.set("id", str(a.id));
	obj.set("name", str(a.name));
	obj.set("description", str(a.description));
	obj.set("hidden", a.hidden);
	if (a.grade != nullptr && *a.grade != '\0') {
		obj.set("grade", a.grade);
	}
	if (a.points != 0) {
		obj.set("points", a.points);
	}
	if (a.rarity_of != nullptr && *a.rarity_of != '\0') {
		Value rarity = Value::object();
		rarity.set("permille", a.rarity_permille);
		rarity.set("of", a.rarity_of);
		obj.set("rarity", std::move(rarity));
	}
	obj.set("dlc", a.dlc);
	obj.set("unlocked_at", a.unlocked ? Value(a.unlocked_at) : Value());
	return obj;
}

Message achievements(const lc_achievement* list, size_t count) {
	Value items = Value::array();
	for (size_t i = 0; i < count; i++) {
		items.push(achievement(list[i]));
	}
	Value obj = typed("achievements");
	obj.set("list", std::move(items));
	return make("achievements", std::move(obj));
}

Message unlocked(const char* id, int64_t at) {
	Value obj = typed("unlocked");
	obj.set("id", str(id));
	obj.set("at", at != LC_UNLOCKED_NOW ? Value(at) : Value());
	return make("unlocked", std::move(obj));
}

Message achievement_progress(const char* id, uint32_t current, uint32_t total) {
	Value obj = typed("achievement_progress");
	obj.set("id", str(id));
	obj.set("current", current);
	obj.set("total", total);
	return make("achievement_progress", std::move(obj));
}

static Value text_or_null(const char* s) {
	return s != nullptr && *s != '\0' ? Value(s) : Value();
}

// A SettingValue: {"Bool": true}, {"Text": "..."}, or {"List": [...]} for a multiple path pick,
// whose lines are its items
static Value setting_value(const lc_setting& s, const char* value) {
	Value obj = Value::object();
	if (s.kind == LC_SETTING_TOGGLE) {
		obj.set("Bool", str(value) == "true");
	} else if (s.kind == LC_SETTING_PATH && s.path != nullptr && s.path->multiple) {
		Value       items = Value::array();
		std::string text  = str(value);
		for (size_t start = 0; start < text.size();) {
			const size_t end = std::min(text.find('\n', start), text.size());
			if (end > start) {
				items.push(text.substr(start, end - start));
			}
			start = end + 1;
		}
		obj.set("List", std::move(items));
	} else {
		obj.set("Text", str(value));
	}
	return obj;
}

static Value setting(const lc_setting& s) {
	Value kind;
	switch (s.kind) {
		case LC_SETTING_TOGGLE: kind = Value("Toggle"); break;
		case LC_SETTING_CHOICE: {
			Value options = Value::array();
			for (size_t i = 0; i < s.option_count; i++) {
				Value option = Value::object();
				option.set("value", str(s.options[i].value));
				option.set("label", str(s.options[i].label));
				options.push(std::move(option));
			}
			Value choice = Value::object();
			choice.set("options", std::move(options));
			kind = Value::object();
			kind.set("Choice", std::move(choice));
			break;
		}
		case LC_SETTING_TEXT: {
			Value text = Value::object();
			text.set("placeholder", str(s.placeholder));
			kind = Value::object();
			kind.set("Text", std::move(text));
			break;
		}
		case LC_SETTING_PATH: {
			const lc_path_pick none {};
			const lc_path_pick& p = s.path != nullptr ? *s.path : none;
			Value pick = Value::object();
			pick.set("pick", p.file ? "file" : "folder");
			pick.set("title", str(p.title));
			pick.set("filters", Value::array());
			pick.set("start", text_or_null(p.start));
			pick.set("places", Value::array());
			pick.set("decorate", false);
			pick.set("space", false);
			pick.set("writable", false);
			pick.set("multiple", p.multiple);
			kind = Value::object();
			kind.set("Path", std::move(pick));
			break;
		}
		default: kind = Value("Action"); break;
	}
	Value obj = Value::object();
	obj.set("key", str(s.key));
	obj.set("group", str(s.group));
	obj.set("page", "");
	obj.set("section", str(s.section));
	obj.set("label", str(s.label));
	obj.set("help", str(s.help));
	obj.set("kind", std::move(kind));
	obj.set("value", setting_value(s, s.value));
	obj.set("default", setting_value(s, s.default_value));
	return obj;
}

Message overlay_items(const lc_setting* items, size_t count) {
	Value list = Value::array();
	for (size_t i = 0; i < count; i++) {
		list.push(setting(items[i]));
	}
	Value obj = typed("overlay_items");
	obj.set("items", std::move(list));
	return make("overlay_items", std::move(obj));
}

Message view(uint32_t width, uint32_t height) {
	Value obj = typed("view");
	obj.set("width", width);
	obj.set("height", height);
	return make("view", std::move(obj));
}

Message capture_surface(uint32_t id, uint32_t width, uint32_t height) {
	Value obj = typed("capture_surface");
	obj.set("id", id);
	obj.set("width", width);
	obj.set("height", height);
	return make("capture_surface", std::move(obj));
}

static const char* osk_kind(lc_osk_kind kind) {
	switch (kind) {
		case LC_OSK_LATIN: return "latin";
		case LC_OSK_URL: return "url";
		case LC_OSK_EMAIL: return "email";
		case LC_OSK_NUMBER: return "number";
		default: return "text";
	}
}

static const char* osk_enter(lc_osk_enter enter) {
	switch (enter) {
		case LC_OSK_ENTER_SEND: return "send";
		case LC_OSK_ENTER_SEARCH: return "search";
		case LC_OSK_ENTER_GO: return "go";
		default: return "default";
	}
}

static const char* osk_align(lc_osk_align align) {
	switch (align) {
		case LC_OSK_ALIGN_CENTER: return "center";
		case LC_OSK_ALIGN_END: return "end";
		default: return "start";
	}
}

// 0 to 1 as thousandths
static uint32_t permille(float f) {
	if (!(f > 0.0f)) {
		return 0;
	}
	return f >= 1.0f ? 1000u : static_cast<uint32_t>(f * 1000.0f + 0.5f);
}

Message osk_open(uint32_t osk, const lc_osk_params& p) {
	Value obj = typed("osk_open");
	obj.set("osk", osk);
	obj.set("kind", osk_kind(p.kind));
	if (p.title != nullptr && *p.title != '\0') {
		obj.set("title", p.title);
	}
	if (p.placeholder != nullptr && *p.placeholder != '\0') {
		obj.set("placeholder", p.placeholder);
	}
	obj.set("text", str(p.text));
	obj.set("cursor", p.cursor);
	if (p.max_length != 0) {
		obj.set("max_length", p.max_length);
	}
	obj.set("multiline", p.multiline);
	obj.set("password", p.password);
	obj.set("enter", osk_enter(p.enter));
	if (p.language_count != 0 && p.languages != nullptr) {
		Value languages = Value::array();
		for (size_t i = 0; i < p.language_count; i++) {
			if (p.languages[i] != nullptr && *p.languages[i] != '\0') {
				languages.push(p.languages[i]);
			}
		}
		obj.set("languages", std::move(languages));
	}
	if (p.has_anchor) {
		Value anchor = Value::object();
		anchor.set("x", permille(p.x));
		anchor.set("y", permille(p.y));
		anchor.set("align_x", osk_align(p.align_x));
		anchor.set("align_y", osk_align(p.align_y));
		obj.set("anchor", std::move(anchor));
	}
	return make("osk_open", std::move(obj));
}

Message osk_text(uint32_t osk, uint64_t seq, const char* text, uint32_t cursor) {
	Value obj = typed("osk_text");
	obj.set("osk", osk);
	obj.set("seq", seq);
	obj.set("text", str(text));
	obj.set("cursor", cursor);
	return make("osk_text", std::move(obj));
}

Message osk_closed(uint32_t osk, lc_osk_result result) {
	Value obj = typed("osk_closed");
	obj.set("osk", osk);
	obj.set("result", result == LC_OSK_ACCEPTED ? "accepted" : result == LC_OSK_CANCELLED ? "cancelled" : "aborted");
	return make("osk_closed", std::move(obj));
}

// 0 to 1 as ten-thousandths
static uint32_t ten_thousandths(float f) {
	if (!(f > 0.0f)) {
		return 0;
	}
	return f >= 1.0f ? 10000u : static_cast<uint32_t>(f * 10000.0f + 0.5f);
}

// Hundredths, rounded, and clamped to what the launcher reads
static int64_t hundredths(float f) {
	const float h = f * 100.0f;
	if (!(h > -2.0e9f && h < 2.0e9f)) {
		return 0;
	}
	return static_cast<int64_t>(h < 0.0f ? h - 0.5f : h + 0.5f);
}

Message overlay_input(const lc_input& in) {
	Value input = Value::object();
	switch (in.kind) {
		case LC_INPUT_KEY:
			if (in.key == nullptr || in.key[0] == '\0') {
				return {};
			}
			input.set("kind", "key");
			input.set("key", in.key);
			input.set("down", in.down);
			input.set("repeat", in.repeat);
			input.set("mods", in.mods & (LC_MOD_SHIFT | LC_MOD_CTRL | LC_MOD_ALT | LC_MOD_SUPER));
			break;
		case LC_INPUT_TEXT:
			if (in.text == nullptr || in.text[0] == '\0') {
				return {};
			}
			input.set("kind", "text");
			input.set("text", str(in.text));
			break;
		case LC_INPUT_POINTER:
			input.set("kind", "pointer");
			input.set("x", ten_thousandths(in.x));
			input.set("y", ten_thousandths(in.y));
			break;
		case LC_INPUT_BUTTON:
			input.set("kind", "button");
			input.set("button", std::min<uint32_t>(in.button, 255));
			input.set("down", in.down);
			break;
		case LC_INPUT_WHEEL:
			input.set("kind", "wheel");
			input.set("x", hundredths(in.x));
			input.set("y", hundredths(in.y));
			input.set("pixels", in.pixels);
			break;
		default: return {};
	}
	Value obj = typed("overlay_input");
	obj.set("input", std::move(input));
	return make("overlay_input", std::move(obj));
}

static Value game(const lc_game& g) {
	Value art = Value::object();
	art.set("cover", text_or_null(g.art.cover));
	art.set("hero", text_or_null(g.art.hero));
	art.set("logo", text_or_null(g.art.logo));
	art.set("icon", text_or_null(g.art.icon));
	Value obj = Value::object();
	obj.set("external_id", str(g.external_id));
	obj.set("title", str(g.title));
	obj.set("platform", str(g.platform));
	obj.set("installed", g.installed);
	obj.set("game_dir", text_or_null(g.game_dir));
	obj.set("art", std::move(art));
	return obj;
}

static Value games(const lc_game* list, size_t count) {
	Value items = Value::array();
	for (size_t i = 0; i < count; i++) {
		items.push(game(list[i]));
	}
	return items;
}

Message library(const lc_game* list, size_t count, const char* const* warnings, size_t warning_count) {
	Value lines = Value::array();
	for (size_t i = 0; i < warning_count; i++) {
		lines.push(str(warnings[i]));
	}
	Value obj = typed("library");
	obj.set("games", games(list, count));
	obj.set("warnings", std::move(lines));
	return make("library", std::move(obj));
}

Message library_changed(const lc_game* list, size_t count) {
	Value obj = typed("library_changed");
	obj.set("games", games(list, count));
	return make("library_changed", std::move(obj));
}

Message launch_spawn(uint64_t launch, const lc_spawn& spawn) {
	Value args = Value::array();
	for (size_t i = 0; i < spawn.arg_count; i++) {
		args.push(str(spawn.args[i]));
	}
	Value env = Value::array();
	for (size_t i = 0; i < spawn.env_count; i++) {
		Value pair = Value::array();
		pair.push(str(spawn.env[i].key));
		pair.push(str(spawn.env[i].value));
		env.push(std::move(pair));
	}
	Value command = Value::object();
	command.set("exe", str(spawn.exe));
	command.set("args", std::move(args));
	command.set("env", std::move(env));
	command.set("cwd", text_or_null(spawn.cwd));
	Value obj = typed("launch_result");
	obj.set("launch", launch);
	obj.set("spawn", std::move(command));
	obj.set("error", Value());
	return make("launch_result", std::move(obj));
}

Message launch_error(uint64_t launch, const char* error) {
	Value obj = typed("launch_result");
	obj.set("launch", launch);
	obj.set("spawn", Value());
	obj.set("error", str(error));
	return make("launch_result", std::move(obj));
}

Message achievement_list(uint64_t game, const lc_achievement* list, size_t count) {
	Value items = Value::array();
	for (size_t i = 0; i < count; i++) {
		items.push(achievement(list[i]));
	}
	Value obj = typed("achievement_list");
	obj.set("game", game);
	obj.set("list", std::move(items));
	return make("achievement_list", std::move(obj));
}

Message settings(const char* page, const lc_setting* items, size_t count) {
	Value list = Value::array();
	for (size_t i = 0; i < count; i++) {
		Value row = setting(items[i]);
		row.set("page", str(page));
		list.push(std::move(row));
	}
	Value obj = typed("settings");
	obj.set("items", std::move(list));
	return make("settings", std::move(obj));
}

Message game_settings(const char* page, const lc_setting* items, size_t count) {
	Value list = Value::array();
	for (size_t i = 0; i < count; i++) {
		Value row = setting(items[i]);
		row.set("page", str(page));
		row.set("live", items[i].live);
		list.push(std::move(row));
	}
	Value obj = typed("game_settings");
	obj.set("items", std::move(list));
	return make("game_settings", std::move(obj));
}

// A SettingValue as text: externally tagged ({"Text": "150"}, {"Bool": true}), or a plain JSON
// value. Null means the default.
static std::optional<std::string> value_text(const Value& value) {
	switch (value.kind()) {
		case Value::Kind::Null: return std::nullopt;
		case Value::Kind::Object:
			if (value.members().size() == 1) {
				return value_text(value.members().front().second);
			}
			return std::nullopt;
		case Value::Kind::String: return value.as_string();
		case Value::Kind::Array: {
			// A list (a multiple path pick's): its items, one per line
			std::string lines;
			for (const auto& item: value.items()) {
				if (item.is_string()) {
					lines += (lines.empty() ? "" : "\n") + item.as_string();
				}
			}
			return lines;
		}
		case Value::Kind::Bool: return value.as_bool() ? "true" : "false";
		case Value::Kind::Int: return std::to_string(value.as_int());
		case Value::Kind::Uint: return std::to_string(value.as_uint());
		case Value::Kind::Float: {
			char buf[32];
			std::snprintf(buf, sizeof(buf), "%g", value.as_float());
			return std::string(buf);
		}
		default: return std::nullopt;
	}
}

static uint64_t read_u64(const Value& obj, const char* name) {
	const Value* v = obj.find(name);
	return v != nullptr && v->is_unsigned() ? v->as_uint() : 0u;
}

static uint32_t read_u32(const Value& obj, const char* name) {
	const Value* v = obj.find(name);
	return v != nullptr && v->is_unsigned() && v->as_uint() <= UINT32_MAX
	           ? static_cast<uint32_t>(v->as_uint())
	           : 0u;
}

HostMessage parse_host(const std::string& frame) {
	HostMessage msg;
	const auto  doc = json::parse(frame);
	if (!doc || !doc->is_object()) {
		msg.ok = false;
		return msg;
	}
	const Value& obj = *doc;
	msg.type         = obj.text("type");
	msg.title        = obj.text("title");
	msg.key          = obj.text("key");
	msg.server       = obj.text("server");
	msg.service      = obj.text("service");
	// Older launchers spell it "Json"
	msg.encoding = obj.text("encoding");
	for (auto& c: msg.encoding) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	msg.header = read_u32(obj, "header");
	msg.width  = read_u32(obj, "width");
	msg.height = read_u32(obj, "height");
	msg.api    = static_cast<int>(read_u32(obj, "api"));
	if (const Value* v = obj.find("buffers"); v != nullptr && v->is_array()) {
		for (const auto& id: v->items()) {
			if (id.is_unsigned() && id.as_uint() <= UINT32_MAX) {
				msg.buffers.push_back(static_cast<uint32_t>(id.as_uint()));
			}
		}
	}
	msg.launch      = read_u64(obj, "launch");
	msg.game        = read_u64(obj, "game");
	msg.external_id = obj.text("external_id");
	msg.game_dir    = obj.text("game_dir");
	if (const Value* v = obj.find("args"); v != nullptr && v->is_array()) {
		for (const auto& arg: v->items()) {
			if (arg.is_string()) {
				msg.args.push_back(arg.as_string());
			}
		}
	}
	if (const Value* v = obj.find("lists"); v != nullptr && v->is_bool()) {
		msg.lists = v->as_bool();
	}
	if (const Value* v = obj.find("visible"); v != nullptr && v->is_bool()) {
		msg.visible = v->as_bool();
	}
	if (const Value* v = obj.find("input"); v != nullptr && v->is_bool()) {
		msg.input = v->as_bool();
	}
	if (const Value* v = obj.find("available"); v != nullptr && v->is_bool()) {
		msg.available = v->as_bool();
	}
	if (const Value* v = obj.find("accepted"); v != nullptr && v->is_bool()) {
		msg.accepted = v->as_bool();
	}
	msg.osk = read_u32(obj, "osk");
	msg.seq = read_u64(obj, "seq");
	if (const Value* edit = obj.find("edit"); edit != nullptr && edit->is_object()) {
		const std::string op = edit->text("op");
		msg.edit_text        = edit->text("text");
		msg.edit_ok          = true;
		if (op == "insert") {
			msg.edit_op = LC_OSK_INSERT;
		} else if (op == "backspace") {
			msg.edit_op = LC_OSK_BACKSPACE;
		} else if (op == "delete") {
			msg.edit_op = LC_OSK_DELETE;
		} else if (const Value* by = edit->find("by"); op == "move" && by != nullptr && by->is_integer() &&
		                                               by->as_int() >= INT32_MIN && by->as_int() <= INT32_MAX) {
			msg.edit_op = LC_OSK_MOVE;
			msg.edit_by = static_cast<int32_t>(by->as_int());
		} else if (op == "replace") {
			msg.edit_op     = LC_OSK_REPLACE;
			msg.edit_cursor = read_u32(*edit, "cursor");
		} else {
			msg.edit_ok = false;
		}
	}
	if (const Value* v = obj.find("value"); v != nullptr) {
		msg.value = value_text(*v);
	}
	if (const Value* v = obj.find("values"); v != nullptr && v->is_object()) {
		for (const auto& [key, value]: v->members()) {
			msg.values.emplace_back(key, value_text(value));
		}
	}
	return msg;
}

} // namespace lc::protocol
