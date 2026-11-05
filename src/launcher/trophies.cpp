// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <fmt/format.h>
#include <pugixml.hpp>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "core/file_format/npbind.h"
#include "core/file_format/trp.h"
#include "core/file_sys/fs.h"
#include "core/file_sys/ifile.h"
#include "launcher/trophies.h"

namespace Launcher::Trophies {

namespace fs = std::filesystem;

namespace {

// The game's own files, then its update's, as a game booting sees them
std::vector<fs::path> Roots(const fs::path& game) {
    std::vector<fs::path> roots;
    for (const auto suffix : {"-UPDATE", "-patch"}) {
        const auto overlay =
            Core::FileSys::OverlayPath(Core::FileSys::StripZArchiveExtension(game), suffix);
        if (const auto root = Core::FileSys::ResolveGameRoot(overlay)) {
            roots.push_back(*root);
            break;
        }
    }
    roots.push_back(game);
    return roots;
}

std::string Grade(std::string_view ttype) {
    if (ttype == "P") {
        return "platinum";
    }
    if (ttype == "G") {
        return "gold";
    }
    if (ttype == "S") {
        return "silver";
    }
    if (ttype == "B") {
        return "bronze";
    }
    return {};
}

fs::path LanguageXml(const fs::path& xml_dir, int language) {
    const auto localized = xml_dir / fmt::format("TROP_{:02}.XML", language);
    std::error_code ec;
    if (language >= 0 && fs::exists(localized, ec)) {
        return localized;
    }
    return xml_dir / "TROP.XML";
}

} // namespace

std::string Id(s32 service_label, s32 trophy_id) {
    if (service_label == 0) {
        return std::to_string(trophy_id);
    }
    return fmt::format("{}:{}", service_label, trophy_id);
}

std::vector<Trophy> List(const std::map<s32, std::string>& sets, s32 user_id, int language) {
    std::vector<Trophy> out;
    for (const auto& [label, np_comm_id] : sets) {
        const auto xml_dir =
            Common::FS::GetUserPath(Common::FS::PathType::TrophyDir) / np_comm_id / "Xml";
        pugi::xml_document conf;
        if (!conf.load_file(LanguageXml(xml_dir, language).native().c_str())) {
            LOG_WARNING(Lib_NpTrophy, "Launcher: no trophy list for {}", np_comm_id);
            continue;
        }
        // The user's copy of TROPCONF.XML holds the unlocks, once the game has booted
        pugi::xml_document saved;
        const auto save_file = EmulatorSettings.GetHomeDir() / std::to_string(user_id) / "trophy" /
                               (np_comm_id + ".xml");
        const bool has_saved = saved.load_file(save_file.native().c_str());

        std::map<s32, s64> unlocks;
        if (has_saved) {
            for (const auto& node : saved.child("trophyconf").children("trophy")) {
                if (node.attribute("unlockstate").as_bool()) {
                    unlocks[node.attribute("id").as_int(-1)] =
                        node.attribute("timestamp").as_llong();
                }
            }
        }

        const std::string set = conf.child("trophyconf").child("title-name").text().as_string();
        for (const auto& node : conf.child("trophyconf").children("trophy")) {
            const s32 id = node.attribute("id").as_int(-1);
            if (id < 0) {
                continue;
            }
            Trophy t;
            t.id = Id(label, id);
            t.name = node.child("name").text().as_string();
            t.description = node.child("detail").text().as_string();
            const std::string_view ttype = node.attribute("ttype").value();
            t.grade = Grade(ttype);
            t.set = set;
            t.hidden = std::string_view(node.attribute("hidden").value()) == "yes";
            // Trophies of a group (gid) come with add-ons
            t.dlc = node.attribute("gid").as_int(-1) >= 0;
            if (const auto it = unlocks.find(id); it != unlocks.end()) {
                t.unlocked_at = it->second;
            }
            out.push_back(std::move(t));
        }
    }
    return out;
}

std::map<s32, std::string> ExtractSets(const fs::path& game) {
    std::map<s32, std::string> sets;
    const auto roots = Roots(game);

    std::optional<std::vector<u8>> npbind_data;
    for (const auto& root : roots) {
        if ((npbind_data = Core::FileSys::ReadGameFile(root, "sce_sys/npbind.dat"))) {
            break;
        }
    }
    NPBindFile npbind;
    if (!npbind_data || !npbind.Load(std::span<const u8>(*npbind_data))) {
        return sets;
    }
    const auto np_comm_ids = npbind.GetNpCommIds();
    const auto trophy_numbers = npbind.GetTrophyNumbers();

    for (size_t index = 0; index < np_comm_ids.size(); index++) {
        const auto& np_comm_id = np_comm_ids[index];
        // The set's file, trophy<number>.trp, whose number is the service label the game opens
        // it with: usually the entry's index, but npbind says (ExtractTrophies)
        const s32 number = trophy_numbers[index];
        const auto output = Common::FS::GetUserPath(Common::FS::PathType::TrophyDir) / np_comm_id;
        std::error_code ec;
        if (fs::exists(output / "Xml", ec)) {
            sets[number] = np_comm_id;
            continue;
        }
        bool extracted = false;
        for (const auto& name :
             {fmt::format("trophy{:02}.trp", number), fmt::format("TROPHY{:02}.TRP", number)}) {
            // An update without trophies falls back to the game's own, as at boot
            for (const auto& root : roots) {
                const auto trp = Core::FileSys::ResolveGameFilePath(root, "sce_sys/trophy/" + name);
                if (!trp) {
                    continue;
                }
                // TRP::Extract takes only a lowercase .trp
                fs::path source = *trp;
                fs::path copy;
                if (source.extension() != ".trp") {
                    copy = fs::temp_directory_path() / (np_comm_id + "_launcher.trp");
                    fs::copy_file(source, copy, fs::copy_options::overwrite_existing, ec);
                    if (ec) {
                        continue;
                    }
                    source = copy;
                }
                TRP extractor;
                extracted = extractor.Extract(source, np_comm_id, output);
                if (!copy.empty()) {
                    fs::remove(copy, ec);
                }
                if (extracted) {
                    break;
                }
            }
            if (extracted) {
                break;
            }
        }
        if (extracted) {
            sets[number] = np_comm_id;
        }
    }
    return sets;
}

} // namespace Launcher::Trophies
