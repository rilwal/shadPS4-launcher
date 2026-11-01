/*
 * launcher_client.h: an emulator's side of the game launcher's plugin protocol
 * (docs/plugin-api.md), as a game session.
 *
 * When launcherd starts a patched emulator it sets LAUNCHER_ENDPOINT (a Unix socket path, or a
 * pipe name on Windows) and LAUNCHER_SESSION (a token for this game session). lc_connect()
 * connects back, says hello and waits for the welcome; after that the emulator reports its status,
 * achievements and pauses with the calls below, and the launcher draws all UI.
 *
 * Messages go out on a writer thread of the library's own, so no emulator thread ever waits on
 * the launcher (lc_failed and lc_flush are the exceptions, and say so). What the launcher asks
 * for arrives through lc_callbacks, on the library's reader thread.
 *
 * Strings are UTF-8 and NUL-terminated; the library copies whatever it keeps. Calls on a NULL
 * session do nothing, so code needn't check for launcher mode at every call.
 *
 * The composited overlay (sharing the shell's quick menu into the game's frames) is in
 * launcher_overlay.h, and its optional Vulkan helper in launcher_overlay_vk.h.
 */
#ifndef LAUNCHER_CLIENT_H
#define LAUNCHER_CLIENT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The plugin API version the library speaks. */
#define LC_PLUGIN_API 4

typedef struct lc_session lc_session;

/* Capabilities beyond "session", which every game session declares. */
enum {
	/* Draws the shell's quick menu into the game's own frames (launcher_overlay.h). Left out of
	   a game session's hello on platforms that can't share surfaces with the launcher. A plugin
	   runner keeps it on every platform: there it says its game sessions draw the menu. */
	LC_CAP_COMPOSITED_OVERLAY = 1u << 0,
	/* A plugin runner's (launcher_runner.h), for lc_runner_connect, which needs at least one of
	   them; lc_connect leaves them out. */
	LC_CAP_LIBRARY      = 1u << 1,
	LC_CAP_LAUNCH       = 1u << 2,
	LC_CAP_ACHIEVEMENTS = 1u << 3,
	LC_CAP_SETTINGS     = 1u << 4,
	/* With LC_CAP_COMPOSITED_OVERLAY: the game's text input can go on the shell's on-screen
	   keyboard, drawn into the game's frames, instead of the emulator's own (lc_osk_open). */
	LC_CAP_OSK = 1u << 5,
};

typedef enum lc_log_level {
	LC_LOG_INFO  = 0,
	LC_LOG_ERROR = 1,
} lc_log_level;

/* A key pressed on the shell's on-screen keyboard (lc_callbacks.osk_edit). Positions count
   characters (Unicode scalar values), whatever the emulator stores text as. */
typedef enum lc_osk_op {
	LC_OSK_INSERT    = 0, /* `text` at the cursor */
	LC_OSK_BACKSPACE = 1, /* The character before the cursor */
	LC_OSK_DELETE    = 2, /* The character after the cursor */
	LC_OSK_MOVE      = 3, /* The cursor, by `by` characters (negative is left) */
	LC_OSK_REPLACE   = 4, /* All the text with `text`, the cursor at `cursor` (a composing keyboard) */
} lc_osk_op;

typedef struct lc_osk_edit {
	lc_osk_op   op;
	const char *text; /* LC_OSK_INSERT, LC_OSK_REPLACE; never NULL */
	int32_t     by;   /* LC_OSK_MOVE */
	uint32_t    cursor; /* LC_OSK_REPLACE */
} lc_osk_edit;

/* What the launcher asks of the game. Every member may be NULL. All but `log` are called on the
   library's reader thread, one at a time; they mustn't call lc_close. */
typedef struct lc_callbacks {
	void *user;

	/* A line for the emulator's log, without a newline. NULL writes it to stderr. Any thread. */
	void (*log)(void *user, lc_log_level level, const char *line);

	/* Save what's needed and exit. launcherd ends the process group 30 s later. */
	void (*quit)(void *user);
	/* The player opened (true) or closed (false) the quick menu. Pause if that suits the game,
	   and say so with lc_paused. With the composited overlay the library also starts or stops
	   the capture itself. */
	void (*overlay)(void *user, bool opened);
	/* The player chose an overlay item: `value` is NULL for an action, "true"/"false" for a
	   toggle, the option's value for a choice. */
	void (*invoke)(void *user, const char *key, const char *value);
	/* A live setting changed while the game runs, by the emulator's own key; `value` is NULL for
	   the default. To a plugin runner: one of its own settings' values, sent for each stored value
	   after lc_settings and on every change. */
	void (*setting)(void *user, const char *key, const char *value);
	/* The connection ended. The process tree is what the launcher tracks, so this isn't fatal. */
	void (*closed)(void *user);

	/* On-screen keyboard (LC_CAP_OSK). Whether the shell can draw the game's keyboards now. When
	   it turns false with one of the shell's open, that keyboard is closed as far as the library
	   is concerned: show the emulator's own instead, with the text as it is. */
	void (*osk_available)(void *user, bool available);
	/* The player pressed a key on keyboard `osk`. Apply it with the emulator's own rules (length,
	   the game's filter), then send the text with lc_osk_text, echoing `seq`. */
	void (*osk_edit)(void *user, uint32_t osk, uint64_t seq, const lc_osk_edit *edit);
	/* The player finished (`accepted`) or backed out of keyboard `osk`'s text. Close it as the
	   emulator's own keyboard would, and say so with lc_osk_close. */
	void (*osk_finish)(void *user, uint32_t osk, bool accepted);
} lc_callbacks;

typedef struct lc_connect_info {
	/* The emulator and its version, shown in the launcher's logs ("Kyty 0.4.1"). */
	const char *name;
	/* LC_CAP_* flags. */
	uint32_t capabilities;
	/* How long to wait for the welcome; 0 means 3000 ms. */
	uint32_t timeout_ms;
} lc_connect_info;

/* Whether the launcher started this process: both environment variables are set. */
bool lc_env_present(void);

/* Connects, sends the hello and waits for the welcome. NULL outside launcher mode or if that
   fails, with the reason in `error` (if not NULL). The session lives until lc_close, or the
   process exits; lc_get returns it meanwhile. */
lc_session *lc_connect(const lc_connect_info *info, const lc_callbacks *callbacks, char *error,
                       size_t error_len);

/* The connected session, or NULL. */
lc_session *lc_get(void);

/* The game's title and the API launcherd chose, from its `accepted`. "" and 0 before it comes. */
const char *lc_title(lc_session *session);
int         lc_api(lc_session *session);

/* One of `configure`'s values. `value` is NULL for the default. */
typedef struct lc_kv {
	const char *key;
	const char *value;
} lc_kv;

/* Waits for `configure`, which launcherd sends right after the welcome and before the game boots:
   the settings the player changed, by the emulator's own keys. Counts `timeout_ms` from
   connecting. False if none came (the connection closed, or the time ran out); a configure that
   comes after this returned is ignored. The values stay valid until lc_close. */
bool lc_wait_configure(lc_session *session, uint32_t timeout_ms, const lc_kv **values,
                       size_t *count);

/* Status */
void lc_booting(lc_session *session);
/* On screen and taking input. `pid` 0 means this process; `window` is a native handle (HWND,
   X11 window id, NSWindow number) or 0 if unknown. Sends the view after it, if one is set. */
void lc_ready(lc_session *session, uint32_t pid, uint64_t window);
/* Boot or run failed; exit after this. `code` (NULL for none) is a snake-case cause such as
   "firmware_missing". Sent once; waits up to 2 s for it to be written. */
void lc_failed(lc_session *session, const char *reason, const char *code);
void lc_paused(lc_session *session, bool paused);
/* Starts or updates something ongoing, such as compiling shaders. `progress` is 0 to 1, or
   negative for none. Updates of one id are sent at most every 100 ms unless the label changes. */
void lc_activity(lc_session *session, const char *id, const char *label, float progress);
void lc_activity_done(lc_session *session, const char *id);
/* What the player is doing, or NULL to clear it. */
void lc_presence(lc_session *session, const char *text);
void lc_notify(lc_session *session, const char *title, const char *body, bool error);

/* Achievements */
typedef struct lc_achievement {
	const char *id;
	const char *name;
	const char *description;
	bool        hidden;
	bool        dlc;
	/* "bronze", "silver", "gold" or "platinum", only where the platform has tiers; else NULL. */
	const char *grade;
	/* Gamerscore and the like; 0 for none. */
	uint32_t points;
	/* Who has it, in thousandths, of `rarity_of` ("owners" or "players"); NULL for unknown. */
	uint32_t    rarity_permille;
	const char *rarity_of;
	/* When it was unlocked (Unix seconds), if `unlocked`. */
	bool    unlocked;
	int64_t unlocked_at;
} lc_achievement;

/* The game's full list, once per session, before unlocks. */
void lc_achievements(lc_session *session, const lc_achievement *list, size_t count);
/* LC_UNLOCKED_NOW for `at` when the emulator doesn't know when; 0 means at an unknown time. */
#define LC_UNLOCKED_NOW INT64_MIN
void lc_unlocked(lc_session *session, const char *id, int64_t at);
void lc_achievement_progress(lc_session *session, const char *id, uint32_t current,
                             uint32_t total);

/* Quick-menu items, and a plugin runner's settings (launcher_runner.h), as the launcher's own
   settings rows */
typedef enum lc_setting_kind {
	LC_SETTING_ACTION = 0, /* A button; its value is ignored */
	LC_SETTING_TOGGLE = 1, /* value "true" or "false" */
	LC_SETTING_CHOICE = 2, /* value is one of `options` */
	LC_SETTING_TEXT   = 3, /* value is free text; `placeholder` shows when it's empty */
	/* A file or folder, chosen with the shell's picker (`path`). value is the path, "" for none; for
	   a `multiple` pick, the paths one per line. A runner's settings only. */
	LC_SETTING_PATH = 4,
} lc_setting_kind;

/* What an LC_SETTING_PATH row's picker picks */
typedef struct lc_path_pick {
	bool        file;     /* A file, else a folder */
	const char *title;    /* What the picker says it's for; NULL uses the label */
	const char *start;    /* Where it opens if there's nothing better; "~" is home. May be NULL */
	bool        multiple; /* A list of them, each choice adding to it */
} lc_path_pick;

typedef struct lc_option {
	const char *value;
	const char *label;
} lc_option;

typedef struct lc_setting {
	const char     *key;
	const char     *section;
	const char     *label;
	const char     *help;
	lc_setting_kind kind;
	const lc_option *options; /* LC_SETTING_CHOICE */
	size_t           option_count;
	const char      *placeholder; /* LC_SETTING_TEXT, may be NULL */
	const char      *value;
	const char      *default_value;
	const lc_path_pick *path; /* LC_SETTING_PATH */
	/* A runner's settings: the group its page is listed under ("Emulators"); NULL or "" for the
	   launcher's default */
	const char *group;
	/* A runner's game settings (lc_game_settings): it can change while a game runs */
	bool live;
} lc_setting;

/* Replaces the quick-menu items. */
void lc_overlay_items(lc_session *session, const lc_setting *items, size_t count);

/* On-screen keyboard (docs/plugin-api.md, "On-screen keyboard"). With LC_CAP_OSK, the shell can
   draw the keyboard a game asks for into the game's frames, as it does the quick menu. The
   emulator keeps the text: the shell sends keys (lc_callbacks.osk_edit), and the emulator
   applies them, along with whatever a real keyboard types, and sends the text after every change
   (lc_osk_text). While the keyboard is open the game keeps running but stops reading the
   controllers, which the shell reads. */

/* Which keys the keyboard needs */
typedef enum lc_osk_kind {
	LC_OSK_TEXT   = 0, /* Anything, in the player's language */
	LC_OSK_LATIN  = 1, /* ASCII letters, digits and marks */
	LC_OSK_URL    = 2,
	LC_OSK_EMAIL  = 3,
	LC_OSK_NUMBER = 4, /* Digits only */
} lc_osk_kind;

/* What the key that finishes the text says */
typedef enum lc_osk_enter {
	LC_OSK_ENTER_DEFAULT = 0, /* "Done" */
	LC_OSK_ENTER_SEND    = 1,
	LC_OSK_ENTER_SEARCH  = 2,
	LC_OSK_ENTER_GO      = 3,
} lc_osk_enter;

typedef enum lc_osk_align {
	LC_OSK_ALIGN_START  = 0,
	LC_OSK_ALIGN_CENTER = 1,
	LC_OSK_ALIGN_END    = 2,
} lc_osk_align;

typedef struct lc_osk_params {
	lc_osk_kind  kind;
	const char  *title;       /* NULL or "" for none */
	const char  *placeholder; /* Shown while the text is empty; NULL or "" for none */
	const char  *text;        /* What's there already; NULL for none */
	uint32_t     cursor;      /* Where in `text`, in characters */
	uint32_t     max_length;  /* In characters; 0 for no limit */
	bool         multiline;
	bool         password;    /* Dots, not what's typed */
	lc_osk_enter enter;
	/* Language tags ("en", "ja") the game takes; none for any */
	const char *const *languages;
	size_t             language_count;
	/* Where the game would have the keyboard: the point (x, y), from 0 to 1 across the window,
	   that its edges (or middle) align_x and align_y go to. Ignored unless has_anchor. */
	bool         has_anchor;
	float        x, y;
	lc_osk_align align_x, align_y;
} lc_osk_params;

typedef enum lc_osk_result {
	LC_OSK_ACCEPTED  = 0, /* The player finished the text */
	LC_OSK_CANCELLED = 1, /* The player backed out of it */
	LC_OSK_ABORTED   = 2, /* The game closed it */
} lc_osk_result;

/* Whether the shell can draw a keyboard into the game now. Any thread. */
bool lc_osk_available(lc_session *session);
/* Asks the shell for a keyboard, returning its number, or 0 if the shell can't draw one now: show
   the emulator's own then. One is open at a time; opening another closes the last (aborted). */
uint32_t lc_osk_open(lc_session *session, const lc_osk_params *params);
/* Keyboard `osk`'s text after a change, whatever made it: the shell's edit `seq` (the last one it
   includes; 0 if none yet), a real keyboard, or the game. `cursor` is in characters. */
void lc_osk_text(lc_session *session, uint32_t osk, uint64_t seq, const char *text, uint32_t cursor);
/* Keyboard `osk` closed: the shell lets its surfaces go. */
void lc_osk_close(lc_session *session, uint32_t osk, lc_osk_result result);

/* Waits until everything queued so far is written, or `timeout_ms` passes. False on timeout. */
bool lc_flush(lc_session *session, uint32_t timeout_ms);

/* Flushes for up to a second, then disconnects and frees the session. */
void lc_close(lc_session *session);

#ifdef __cplusplus
}
#endif

#endif /* LAUNCHER_CLIENT_H */
