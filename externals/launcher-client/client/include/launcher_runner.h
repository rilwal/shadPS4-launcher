/*
 * launcher_runner.h: an emulator as a plugin runner (docs/plugin-api.md, "Plugin runner
 * messages"), so the emulator itself is the launcher's plugin for its games.
 *
 * launcherd starts the plugin from its config and keeps it running. The plugin connects with
 * lc_runner_connect, then finds the games (lc_library), says how to start one (lc_launch_spawn,
 * usually its own executable with the game's folder), lists a game's achievements without playing
 * it (lc_achievement_list) and describes its settings (lc_settings). The game process that
 * launcherd then spawns connects back as an ordinary game session (launcher_client.h).
 *
 * It's the same connection as a session underneath: the lc_session calls that make sense here
 * (lc_notify, lc_flush, lc_close) work on it too. lc_callbacks.setting brings the values of its
 * settings, and lc_callbacks.invoke a press of one of its LC_SETTING_ACTION rows (`value` NULL).
 * What the launcher asks of a runner arrives through lc_runner_callbacks, on the library's reader
 * thread, one at a time, so a slow one holds up the rest (settings too): hand long work, such as
 * a scan, to a thread of the runner's own. Answer from the callback or later from any thread, but
 * the strings a callback is given are valid only during the call: copy what an answer needs.
 *
 * Covered: library, launch answered with a spawn, achievements, global settings (folders of games
 * among them, as an LC_SETTING_PATH row), and settings for the games it starts (lc_game_settings).
 * Not covered (the runner declares none of them): starting a game itself (launch.session,
 * for_game, processes, exited, stop), install and its tasks, a store's play time, secret text
 * settings, decorating the picker's folders.
 */
#ifndef LAUNCHER_RUNNER_H
#define LAUNCHER_RUNNER_H

#include "launcher_client.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A launch to answer. Its strings are valid only during the launch callback. */
typedef struct lc_launch_request {
	/* Answer with this, within 30 seconds */
	uint64_t    launch;
	uint64_t    game;
	const char *external_id;
	const char *game_dir; /* "" if the game has none */
	/* The player's per-game override of the arguments, if set */
	const char *const *args;
	size_t             arg_count;
} lc_launch_request;

/* What launcherd asks a runner. Every member may be NULL. */
typedef struct lc_runner_callbacks {
	void *user;
	/* Look again and answer with lc_library. */
	void (*scan)(void *user);
	/* Start this game: answer with lc_launch_spawn or lc_launch_error. */
	void (*launch)(void *user, const lc_launch_request *request);
	/* List this game's achievements, without it running: answer with lc_achievement_list.
	   `external_id` and `game_dir` are valid only during the call. */
	void (*list_achievements)(void *user, uint64_t game, const char *external_id,
	                          const char *game_dir);
} lc_runner_callbacks;

/* Connects as a plugin runner. `info->capabilities` holds LC_CAP_LIBRARY and the rest of what it
   provides; without any runner capability it fails. LC_CAP_COMPOSITED_OVERLAY here says the game
   sessions it spawns draw the quick menu themselves, so launcherd adds no overlay hook of its own to
   their launches (the macOS Metal hook). NULL outside launcher mode or if that fails, as
   lc_connect. */
lc_session *lc_runner_connect(const lc_connect_info *info, const lc_callbacks *callbacks,
                              const lc_runner_callbacks *runner, char *error, size_t error_len);

/* Picture files on this machine; NULL for a slot the game has no picture for. */
typedef struct lc_art {
	const char *cover;
	const char *hero;
	const char *logo;
	const char *icon;
} lc_art;

typedef struct lc_game {
	/* Unique in this runner: a title id for a console game */
	const char *external_id;
	const char *title;
	/* "Ps3", "Ps4", "Ps5" or "Pc" */
	const char *platform;
	bool        installed;
	const char *game_dir; /* NULL if it isn't on disk; passed back in lc_launch_request */
	lc_art      art;
} lc_game;

/* Every game the runner knows, answering scan or on its own (send one on connecting). Games left
   out stay in the launcher's history as not installed. `warnings` show with the runner. */
void lc_library(lc_session *session, const lc_game *games, size_t count,
                const char *const *warnings, size_t warning_count);
/* Adds or updates just these games. */
void lc_library_changed(lc_session *session, const lc_game *games, size_t count);

typedef struct lc_spawn {
	const char        *exe;
	const char *const *args;
	size_t             arg_count;
	const lc_kv       *env; /* Added to launcherd's own environment */
	size_t             env_count;
	const char        *cwd; /* NULL leaves launcherd's own working directory */
} lc_spawn;

/* Answers a launch: launcherd runs this command with LAUNCHER_SESSION set and tracks it. */
void lc_launch_spawn(lc_session *session, uint64_t launch, const lc_spawn *spawn);
/* Answers a launch: nothing starts, and the player sees `error`. */
void lc_launch_error(lc_session *session, uint64_t launch, const char *error);

/* Answers list_achievements: the game's full list. Unlocks the launcher already has are kept. */
void lc_achievement_list(lc_session *session, uint64_t game, const lc_achievement *list,
                         size_t count);

/* Describes the runner's global settings. The launcher shows them on a page named `page` (under
   each row's `group`), stores them, and sends each value with lc_callbacks.setting. */
void lc_settings(lc_session *session, const char *page, const lc_setting *items, size_t count);

/* Describes settings for the games it starts, such as an emulator's: toggles and choices, with
   their defaults. The launcher shows them on the page named `page` and, as "Global (...)" choices,
   on each game's page, and stores them. Each game session the runner has launcherd spawn gets
   the ones the player changed (for that game, else globally) in its configure (lc_wait_configure),
   by these keys, and a `live` one's changes while it runs as lc_callbacks.setting. The runner itself
   isn't sent their values. launcherd leaves out rows that aren't toggles or choices, choices with
   an option valued "global" (its own), and keys the runner's lc_settings already uses. */
void lc_game_settings(lc_session *session, const char *page, const lc_setting *items, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* LAUNCHER_RUNNER_H */
