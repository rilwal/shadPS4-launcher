// The plugin messages the library builds and reads (docs/plugin-api.md). The launcher writes an
// example of every message from its Rust types into docs/plugin-api/examples, and the tests check
// these against them, so the two can't drift apart unnoticed.
#ifndef LAUNCHER_CLIENT_PROTOCOL_H
#define LAUNCHER_CLIENT_PROTOCOL_H

#include "launcher_client.h"
#include "launcher_overlay.h"
#include "launcher_runner.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lc::protocol {

// The capabilities that make a connection a plugin runner rather than a game session
constexpr uint32_t LC_CAP_RUNNER = LC_CAP_LIBRARY | LC_CAP_LAUNCH | LC_CAP_ACHIEVEMENTS | LC_CAP_SETTINGS;

// An encoded message, ready to send
struct Message {
	std::string type;
	std::string json;
	std::string activity_id; // activity and activity_done only, for coalescing updates
};

// Plugin to launcher
Message hello(const std::string& token, int api, uint32_t capabilities, const std::string& name);
Message booting();
Message ready(uint32_t pid, uint64_t window);
Message failed(const char* reason, const char* code);
Message paused(bool paused);
Message activity(const char* id, const char* label, float progress);
Message activity_done(const char* id);
Message presence(const char* text);
Message notify(const char* title, const char* body, bool error);
Message achievements(const lc_achievement* list, size_t count);
Message unlocked(const char* id, int64_t at);
Message achievement_progress(const char* id, uint32_t current, uint32_t total);
Message overlay_items(const lc_setting* items, size_t count);
Message view(uint32_t width, uint32_t height);
Message capture_surface(uint32_t id, uint32_t width, uint32_t height);
Message osk_open(uint32_t osk, const lc_osk_params& params);
Message osk_text(uint32_t osk, uint64_t seq, const char* text, uint32_t cursor);
Message osk_closed(uint32_t osk, lc_osk_result result);
// Empty type if there's nothing to send (an unknown kind, or a key with no name)
Message overlay_input(const lc_input& input);

// Plugin runner to launcher
Message library(const lc_game* games, size_t count, const char* const* warnings, size_t warning_count);
Message library_changed(const lc_game* games, size_t count);
Message launch_spawn(uint64_t launch, const lc_spawn& spawn);
Message launch_error(uint64_t launch, const char* error);
Message achievement_list(uint64_t game, const lc_achievement* list, size_t count);
Message settings(const char* page, const lc_setting* items, size_t count);
Message game_settings(const char* page, const lc_setting* items, size_t count);

using Values = std::vector<std::pair<std::string, std::optional<std::string>>>;

// A launcher message, with the fields the library reads. Absent fields read as empty.
struct HostMessage {
	bool        ok = true; // Valid JSON, an object
	std::string type;
	// welcome
	std::string encoding;
	std::string server;
	// accepted
	int         api = 0;
	std::string title;
	// invoke, setting
	std::string                key;
	std::optional<std::string> value;
	// configure
	Values values;
	// overlay_surface
	std::vector<uint32_t> buffers;
	uint32_t              header = 0;
	std::string           service;
	uint32_t              width  = 0;
	uint32_t              height = 0;
	bool                  lists  = false; // The buffers are draw-list slots
	// overlay_visible
	bool visible = false;
	// osk_available, osk_edit, osk_finish
	bool        available = false;
	uint32_t    osk       = 0;
	uint64_t    seq       = 0;
	bool        accepted  = false;
	bool        edit_ok   = false; // osk_edit's edit is one the library knows
	lc_osk_op   edit_op   = LC_OSK_INSERT;
	std::string edit_text;
	int32_t     edit_by     = 0;
	uint32_t    edit_cursor = 0;
	// launch, list_achievements
	uint64_t                 launch = 0;
	uint64_t                 game   = 0;
	std::string              external_id;
	std::string              game_dir;
	std::vector<std::string> args;
};

HostMessage parse_host(const std::string& frame);

} // namespace lc::protocol

#endif // LAUNCHER_CLIENT_PROTOCOL_H
