// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

union SDL_Event;
struct SDL_Window;

// While the launcher's quick menu or keyboard shows in the game's frames, the game's window keeps
// the focus, so its keys and mouse are passed on to the launcher (lc_overlay_input) instead of the
// game. Outside launcher mode, and while the overlay isn't shown, nothing here does anything.
namespace Launcher::Input {

// Whether the launcher has the window's keys and mouse now. Any thread.
[[nodiscard]] bool Wanted();

// On the main thread, for each event before the game sees it: passes it on if the launcher has
// the keys and mouse, and returns true if the game isn't to see it. Key and button releases still
// reach the game, so a key held as the menu opened is let go.
bool Forward(SDL_Window* window, const SDL_Event& event);

} // namespace Launcher::Input
