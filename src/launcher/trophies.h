// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common/types.h"

// A game's trophies as the launcher's achievements: listed by the plugin before the game is ever
// played, and by a game session once the game opens its trophy set.
namespace Launcher::Trophies {

struct Trophy {
    // The trophy's id, prefixed with its set's service label unless that's 0 ("1:12")
    std::string id;
    std::string name;
    std::string description;
    std::string grade; // "bronze", "silver", "gold", "platinum"
    bool hidden = false;
    bool dlc = false;
    std::optional<s64> unlocked_at; // Unix seconds
};

// The launcher's id for a trophy of a set.
[[nodiscard]] std::string Id(s32 service_label, s32 trophy_id);

// Every trophy of a game's sets (service label -> NP communication id), as extracted to shadPS4's
// trophy folder, in `language`, with the unlocks shadPS4 keeps for `user_id`.
[[nodiscard]] std::vector<Trophy> List(const std::map<s32, std::string>& sets, s32 user_id,
                                       int language);

// A game's trophy sets from its sce_sys/npbind.dat, extracting each set's trophy file (TRP) to
// shadPS4's trophy folder unless it's there already, as shadPS4 does when the game boots. Works
// on a game's folder or its .zar archive. Empty without trophies, or without the trophy key.
[[nodiscard]] std::map<s32, std::string> ExtractSets(const std::filesystem::path& game);

} // namespace Launcher::Trophies
