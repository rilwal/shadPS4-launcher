// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "common/types.h"

// The PS4 trophy key the player brings, installed from a file they pick on shadPS4's settings
// page in the game launcher. shadPS4 keeps it in keys.json (KeyManager), as its own launcher does.
namespace Launcher::Keys {

// The trophy key in a file: shadPS4's keys.json, an older shadPS4 config.toml ([Keys] TrophyKey),
// a text file with the key as 32 hex digits, or the 16-byte key itself. Else why not.
[[nodiscard]] std::variant<std::vector<u8>, std::string> ReadTrophyKey(
    const std::filesystem::path& file);

// Reads the trophy key from `file` and saves it to keys.json, where the games shadPS4 starts read
// it. Returns an error, or an empty string.
std::string InstallTrophyKey(const std::filesystem::path& file);

// Whether shadPS4 has a trophy key.
[[nodiscard]] bool HaveTrophyKey();

} // namespace Launcher::Keys
