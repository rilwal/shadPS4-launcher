<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Launcher mode

This fork of shadPS4 works with the [game launcher](https://github.com/rilwal/game-launcher) as a
patched emulator (the launcher's design decision D26). The launcher draws all UI; shadPS4 reports
to it over the launcher's plugin protocol (`docs/plugin-api.md` there), through the launcher's
client library, vendored in `externals/launcher-client`.

shadPS4 is the launcher's plugin for its own games: the launcher has no shadPS4 code of its own.

```toml
[plugins.shadps4]
exe = "/path/to/shadps4"
[plugins.shadps4.settings]
game_dirs = ["~/Games/PS4"]
```

## The plugin

Started with no arguments (or `--launcher-plugin [--game-dir <folders>]`) and the launcher's
environment, `shadps4` is a plugin runner (`src/launcher/plugin.cpp`):

- `library`: the games in its game folders and in shadPS4's own (its config's enabled game
  install folders), each a game's folder or `.zar` archive, or a folder of them, two levels deep.
  A game's update (`-UPDATE` or `-patch` next to it) gives its title and pictures when it has
  them, as when the game boots: the icon (`icon0.png`) and the background (`pic1.png`).
- `launch`: answered with a `spawn` of itself, `--game <game>`, in the folder the plugin runs in,
  so the game finds the same user folder (a portable `user` folder there, or the system one). A
  player's per-game arguments replace those, with `{game_dir}`, `{boot}` and `{title_id}` filled
  in.
- `achievements`: a game's trophies and the default user's unlocks, before it's ever played. The
  trophy files are extracted to shadPS4's trophy folder as a booting game would, so this needs the
  trophy key in shadPS4's keys file.
- `settings` and `game_settings`, below.

The plugin logs to `shadps4_plugin.log` in shadPS4's log folder.

## How a game starts

The plugin has launcherd start `shadps4 --game <game>` with two environment variables:

- `LAUNCHER_ENDPOINT`: the launcher's socket (Unix) or named pipe (Windows).
- `LAUNCHER_SESSION`: a token for this game session.

When both are set, shadPS4 connects before it boots (`src/launcher/session.cpp`). Without them,
nothing changes and shadPS4 runs as before.

## What shadPS4 sends and does

| When | Message |
| --- | --- |
| Connected | `booting` |
| First game frame on screen | `ready {pid, window}`: the HWND, X11 window id or NSWindow number (0 on Wayland) |
| Settings the launcher sent can't be applied | `notify` (error) |
| The game opens its trophy set | `achievements`: every trophy of the game's sets with the default user's unlocks |
| A trophy (and maybe the platinum) unlocks | `unlocked {id, at}` per trophy |
| An assertion fails | `failed {reason, code: "fatal_error"}` |
| The game can't be found or loaded | `failed {code: "invalid_file_or_folder"}` |
| `quit` | the window closes as if the player closed it, and shadPS4 exits |

In launcher mode shadPS4's own trophy pop-ups are off, since the launcher shows them.

## The quick menu

On Linux the launcher's quick menu is drawn in the game's own window (the launcher's composited
overlay, `src/launcher/overlay.cpp`, through the client library's Vulkan helper):

- The game session and the plugin say they draw the menu (`composited_overlay`), so launcherd adds
  no overlay of its own. In launcher mode shadPS4 enables the device extensions the helper needs
  for sharing images with the launcher (dma-bufs), when the GPU has them.
- The shell draws the menu into buffers it shares with shadPS4, and the presenter draws the newest
  one over every frame it presents, after shadPS4's own overlays.
- When the menu opens, shadPS4 pauses the game's threads (as its own pause key does) and reports
  `paused`. The present thread keeps showing the frame it paused on, and the first of those
  presents copies that frame, without shadPS4's own overlays, for the menu's background. While
  nothing else presents (loading), shadPS4 keeps presenting for as long as the menu needs it.
- The pad is the menu's while it's open: the game reads the state it had when the menu opened and
  no new samples. Buttons still held when the menu closes stay hidden from the game until they're
  released, so the press that closed the menu doesn't reach it.
- Closing the menu resumes the game, unless the player had paused it already.

If the GPU can't share images with the launcher, shadPS4 logs why, and the launcher's own window
comes to the front for the menu, as for any game. Elsewhere (macOS, Windows) that's how the menu
shows for now.

Trophy ids are the trophy ids as text, prefixed with the service label for any trophy set past
the first (`"1:12"`), as the Kyty fork does. Only the default user's trophies are reported.

## Settings

The options the launcher shows are in `src/launcher/settings.cpp`, on shadPS4's page under
Emulators. Each one's default is what shadPS4's own config says, so a player who never changes it
in the launcher keeps shadPS4's value (and its per-game config).

- The plugin's own settings: the game folders (the launcher's folder picker) and the console
  language, which the plugin lists trophies in. It passes the language to the games it starts as
  `--launcher-setting console-language=<n>`.
- Everything else is a game setting (`game_settings`): toggles and choices, numbers being offered
  as steps. The launcher stores the player's values, globally and per game, and sends them with
  `configure` before boot. shadPS4 applies them after loading its config for the game, as
  game-specific values over both its global and per-game config, and never saves them.

Some can change while the game runs: the launcher sends the change (`setting`), and shadPS4
applies it at once. Going back to the default means going back to what shadPS4's config said when
the game booted.

| Live | How it takes effect |
| --- | --- |
| Window size, display mode | The window follows on the main thread (a `SDL_EVENT_LAUNCHER_WINDOW` event saying which changed). SDL ignores a size set while the window is fullscreen or maximized, so a new size waits until it leaves fullscreen, and a maximized window is restored to take it |
| Present mode | The swapchain is made again on the next present |
| FSR, RCAS, sharpening strength | The presenter's upscaling settings |
| Frame rate | shadPS4's frame rate counter |
| Volume | The audio output's gain |
| Hide the cursor | Read where it's used |
| Input in the background | SDL's background joystick hint |

The rest (output resolution, HDR, vblank frequency, the accuracy options, PS4 Pro mode, the
network, the audio backend, Confirm with Circle, motion controls (the sensors are turned on when a
controller connects), the console language) are read while the game
boots; their help says they take effect when the game next starts, and the launcher keeps a change
for the next boot. shadPS4's own in-game settings dialog reloads its config when it closes, after
which the launcher's values, live changes included, are applied again.

## The on-screen keyboard

When a game asks for text (`sceImeDialogInit`), the launcher draws its own keyboard into the game's
window instead of shadPS4's, wherever it can draw the quick menu there (`LC_CAP_OSK`, Linux for
now). `src/launcher/osk.cpp` bridges the client library's keyboard calls to the IME dialog
(`ImeDialogUi::LauncherKeyboard` in `src/core/libraries/ime/ime_dialog_ui.cpp`):

- The dialog asks for a keyboard the first time it draws, with the game's title, placeholder, text,
  caret, length limit, kind (text, Latin, URL, email, number), multiline, password, Enter label and
  panel position.
- The dialog keeps the text. Each of the launcher's edits (insert, backspace, delete, move, replace;
  positions in characters, where the dialog counts UTF-16 units) is applied to it within the length
  limit and copied to the game's buffer; the game's text filter runs on the game's thread, as for
  shadPS4's own keyboard. Whenever the text changes, it goes back to the launcher with the last edit
  it includes.
- Done or Back in the launcher's keyboard finishes the dialog as the game's own buttons would (OK,
  or cancelled with the text it started with). The button that did it is hidden from the game
  until it's let go. The game closing the dialog closes the launcher's keyboard (aborted).
- While the launcher can't draw keyboards, or stops being able to, shadPS4's own keyboard is
  drawn, with the text as it is.

The game doesn't read the pads while the dialog is open, either way.

## Not done yet

- The quick menu and keyboard in the game's window on macOS and Windows.
- The launcher's keyboard for `sceIme` (the keyboard games draw themselves, without the dialog),
  and typing on a real keyboard while the launcher's keyboard is up.
- The console language is global only, since the plugin needs it for trophy lists. Trophy lists
  aren't sent again when the language changes.
- A game that restarts itself (`sceSystemServiceLoadExec`) starts a new process, which connects
  again with the same session.
