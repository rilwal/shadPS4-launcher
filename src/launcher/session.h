// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"

// A game the game launcher started: launcherd runs `shadps4 --game <game>` with LAUNCHER_ENDPOINT
// and LAUNCHER_SESSION set, and shadPS4 reports to it through the launcher's client library
// (launcher_client.h). Without those variables every call here does nothing.
namespace Launcher::Session {

// Connects, says the game is booting and waits for the settings launcherd sends before boot.
// `settings` are the plugin's own ("key=value", from --launcher-setting).
void Connect(std::vector<std::string> settings);

[[nodiscard]] bool Active();

// Once shadPS4 has loaded its config for the game: the launcher's values over it, in memory.
// shadPS4's own trophy toasts are turned off, as the launcher shows them.
void ApplySettings();

// The game's first frame is on screen.
void FirstFrame();

// Boot or the game failed; shadPS4 exits after this. `code` is a snake-case cause, or nullptr.
void Failed(std::string_view reason, const char* code = nullptr);

// The game opened a trophy set for `user_id`: the first time, every trophy of the game's sets
// (service label -> NP communication id) with their stored unlocks. Only the default user's.
void TrophiesOpened(const std::map<s32, std::string>& sets, s32 user_id);

// A trophy unlocked for `user_id`, at `at` (Unix seconds).
void TrophyUnlocked(s32 user_id, s32 service_label, s32 trophy_id, u64 at);

// Flushes what's queued, before the process exits.
void Close();

} // namespace Launcher::Session
