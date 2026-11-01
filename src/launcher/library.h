// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace Launcher::Library {

struct Game {
    std::string title_id;
    std::string title;
    // The game's folder, or its .zar archive: what shadPS4 boots
    std::filesystem::path root;
    std::filesystem::path icon; // sce_sys/icon0.png
    std::filesystem::path hero; // sce_sys/pic1.png
};

struct Scan {
    std::vector<Game> games;
    std::vector<std::string> warnings;
};

// The games in `folders`: each a game's folder or archive, or a folder of them, up to two levels
// deep. Updates ("-UPDATE", "-patch") and mods ("-mods") are left out as games of their own, but
// an update's title and pictures win over the game's. The first copy of a title id wins.
[[nodiscard]] Scan Find(const std::vector<std::filesystem::path>& folders);

} // namespace Launcher::Library
