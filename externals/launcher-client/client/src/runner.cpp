// The plugin runner's calls (launcher_runner.h). Connecting is lc_connect's, in session.cpp.
#include "internal.h"

using namespace lc;

extern "C" {

void lc_library(lc_session* session, const lc_game* games, size_t count, const char* const* warnings, size_t warning_count) {
	if (session != nullptr) {
		session->send(protocol::library(games, count, warnings, warning_count));
	}
}

void lc_library_changed(lc_session* session, const lc_game* games, size_t count) {
	if (session != nullptr) {
		session->send(protocol::library_changed(games, count));
	}
}

void lc_launch_spawn(lc_session* session, uint64_t launch, const lc_spawn* spawn) {
	if (session != nullptr && spawn != nullptr) {
		session->send(protocol::launch_spawn(launch, *spawn));
	}
}

void lc_launch_error(lc_session* session, uint64_t launch, const char* error) {
	if (session != nullptr) {
		session->send(protocol::launch_error(launch, error));
	}
}

void lc_achievement_list(lc_session* session, uint64_t game, const lc_achievement* list, size_t count) {
	if (session != nullptr) {
		session->send(protocol::achievement_list(game, list, count));
	}
}

void lc_settings(lc_session* session, const char* page, const lc_setting* items, size_t count) {
	if (session != nullptr) {
		session->send(protocol::settings(page, items, count));
	}
}

void lc_game_settings(lc_session* session, const char* page, const lc_setting* items, size_t count) {
	if (session != nullptr) {
		session->send(protocol::game_settings(page, items, count));
	}
}

} // extern "C"
