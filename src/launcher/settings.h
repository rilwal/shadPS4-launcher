// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// shadPS4's options as rows of its page in the game launcher's settings. Keys are short
// kebab-case names ("present-mode"). The launcher stores the player's values and hands them to a
// game at boot; they're applied over shadPS4's own config in memory and never saved to it.
namespace Launcher::Settings {

struct Row {
    enum class Kind { Toggle, Choice };
    std::string key;
    std::string label;
    std::string section;
    std::string help;
    Kind kind = Kind::Toggle;
    std::vector<std::pair<std::string, std::string>> options; // Choice: value, label
    // What shadPS4's own config says (EmulatorSettings as loaded), offered as the default
    std::string default_value;
    // The plugin's own setting, passed to the games it starts with SettingFlag, rather than one
    // the launcher sends in configure
    bool runner = false;
};

// Every row, in order, with defaults read from EmulatorSettings.
[[nodiscard]] std::vector<Row> Rows();

// Applies a value over the loaded config, as a game-specific value that's never saved. No value
// leaves the loaded config's own. Returns an error, or an empty string.
std::string Apply(std::string_view key, const std::optional<std::string>& value);

// What a spawned game gets on its command line for one of the plugin's own settings, as
// "key=value".
inline constexpr char SettingFlag[] = "--launcher-setting";

} // namespace Launcher::Settings
