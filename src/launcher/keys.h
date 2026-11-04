// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common/types.h"

// The PS4 trophy key the player types on shadPS4's settings page in the game launcher. shadPS4
// keeps it in keys.json (KeyManager), as its own launcher does.
namespace Launcher::Keys {

// The 128-bit key as 32 hex digits, in either case; spaces are ignored. Returns the key or an
// error for the player.
[[nodiscard]] std::variant<std::vector<u8>, std::string> ParseTrophyKey(std::string_view text);

// The key shadPS4 has; empty if it has none.
[[nodiscard]] std::vector<u8> TrophyKey();

// The key shadPS4 has, as 32 hex digits; empty if it has none.
[[nodiscard]] std::string TrophyKeyHex();

// Saves the key to keys.json, where the games shadPS4 starts read it. Returns an error, or an
// empty string.
std::string SaveTrophyKey(const std::vector<u8>& key);

// Whether shadPS4 has a trophy key.
[[nodiscard]] bool HaveTrophyKey();

} // namespace Launcher::Keys
