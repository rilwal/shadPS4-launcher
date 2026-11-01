// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <set>
#include <fmt/format.h>

#include "common/path_util.h"
#include "core/file_format/psf.h"
#include "core/file_sys/fs.h"
#include "core/file_sys/ifile.h"
#include "launcher/library.h"

namespace Launcher::Library {

namespace fs = std::filesystem;

namespace {

constexpr int MAX_DEPTH = 2;

bool IsOverlay(const fs::path& path) {
    return Core::FileSys::BaseGameFromOverlay(path).has_value();
}

bool IsGameRoot(const fs::path& path) {
    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        return fs::is_regular_file(path / "sce_sys" / "param.sfo", ec) &&
               fs::is_regular_file(path / "eboot.bin", ec);
    }
    return Core::FileSys::IsZArchiveFile(path) &&
           Core::FileSys::ReadGameFile(path, "sce_sys/param.sfo").has_value();
}

// A file of the game as it boots: from its update if that has it
std::optional<fs::path> GameFile(const fs::path& root, std::string_view rel) {
    const auto stem = Core::FileSys::StripZArchiveExtension(root);
    for (const auto suffix : {"-UPDATE", "-patch"}) {
        if (const auto overlay =
                Core::FileSys::ResolveGameRoot(Core::FileSys::OverlayPath(stem, suffix))) {
            if (auto path = Core::FileSys::ResolveGameFilePath(*overlay, rel)) {
                return path;
            }
        }
    }
    return Core::FileSys::ResolveGameFilePath(root, rel);
}

std::optional<Game> ReadGame(const fs::path& root) {
    const auto sfo_path = GameFile(root, "sce_sys/param.sfo");
    PSF sfo;
    if (!sfo_path || !sfo.Open(*sfo_path)) {
        return std::nullopt;
    }
    Game game;
    game.root = root;
    game.title_id = std::string(sfo.GetString("TITLE_ID").value_or(""));
    game.title = std::string(sfo.GetString("TITLE").value_or(""));
    if (game.title_id.empty()) {
        return std::nullopt;
    }
    if (game.title.empty()) {
        game.title = game.title_id;
    }
    game.icon = GameFile(root, "sce_sys/icon0.png").value_or(fs::path{});
    game.hero = GameFile(root, "sce_sys/pic1.png").value_or(fs::path{});
    return game;
}

void ScanFolder(const fs::path& folder, int depth, Scan& scan, std::set<std::string>& seen) {
    if (IsGameRoot(folder)) {
        if (IsOverlay(folder)) {
            return;
        }
        if (auto game = ReadGame(folder)) {
            if (seen.insert(game->title_id).second) {
                scan.games.push_back(std::move(*game));
            }
        } else {
            scan.warnings.push_back(
                fmt::format("Can't read the game in {}", Common::FS::PathToUTF8String(folder)));
        }
        return;
    }
    if (depth >= MAX_DEPTH) {
        return;
    }
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) {
        return;
    }
    std::vector<fs::path> entries;
    for (fs::directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec)) {
        entries.push_back(it->path());
    }
    std::ranges::sort(entries);
    for (const auto& entry : entries) {
        if (!IsOverlay(entry)) {
            ScanFolder(entry, depth + 1, scan, seen);
        }
    }
}

} // namespace

Scan Find(const std::vector<fs::path>& folders) {
    Scan scan;
    std::set<std::string> seen;
    for (const auto& folder : folders) {
        std::error_code ec;
        if (!fs::exists(folder, ec)) {
            scan.warnings.push_back(
                fmt::format("{} doesn't exist", Common::FS::PathToUTF8String(folder)));
            continue;
        }
        ScanFolder(folder, 0, scan, seen);
    }
    return scan;
}

} // namespace Launcher::Library
