// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "common/key_manager.h"
#include "common/path_util.h"
#include "launcher/keys.h"

namespace Launcher::Keys {

namespace fs = std::filesystem;

namespace {

constexpr size_t KEY_SIZE = 16;
// Bigger than any key file or config
constexpr uintmax_t MOST = 1 << 20;

bool IsHex(char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

std::vector<u8> FromHex(std::string_view hex) {
    std::vector<u8> bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        bytes.push_back(static_cast<u8>(std::stoi(std::string(hex.substr(i, 2)), nullptr, 16)));
    }
    return bytes;
}

// shadPS4's keys.json holds it as hex (key_manager.h)
std::optional<std::vector<u8>> FromJson(const std::string& text) {
    const auto j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("TrophyKeySet")) {
        return std::nullopt;
    }
    const auto& set = j.at("TrophyKeySet");
    if (!set.is_object() || !set.contains("ReleaseTrophyKey") ||
        !set.at("ReleaseTrophyKey").is_string()) {
        return std::vector<u8>{};
    }
    const auto hex = set.at("ReleaseTrophyKey").get<std::string>();
    if (!std::ranges::all_of(hex, IsHex) || hex.size() % 2 != 0) {
        return std::vector<u8>{};
    }
    return FromHex(hex);
}

} // namespace

std::variant<std::vector<u8>, std::string> ReadTrophyKey(const fs::path& file) {
    const auto name = Common::FS::PathToUTF8String(file.filename());
    std::error_code ec;
    const auto size = fs::file_size(file, ec);
    if (ec) {
        return fmt::format("Couldn't read {}: {}", name, ec.message());
    }
    if (size > MOST) {
        return fmt::format("{} is too big to be a key file.", name);
    }
    std::ifstream in(file, std::ios::binary);
    std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (!in.good() && !in.eof()) {
        return fmt::format("Couldn't read {}.", name);
    }

    // The key itself, as bytes
    if (text.size() == KEY_SIZE) {
        return std::vector<u8>(text.begin(), text.end());
    }

    if (const auto key = FromJson(text)) {
        if (key->empty()) {
            return fmt::format("There's no trophy key in {}.", name);
        }
        if (key->size() != KEY_SIZE) {
            return fmt::format("The trophy key in {} isn't 16 bytes long.", name);
        }
        return *key;
    }

    // In text (a config.toml's TrophyKey, or a file of its own): the one run of 32 hex digits
    std::set<std::string> found;
    for (size_t at = 0; at < text.size();) {
        if (!IsHex(text[at])) {
            at++;
            continue;
        }
        const auto end = std::find_if_not(text.begin() + at, text.end(), IsHex) - text.begin();
        if (static_cast<size_t>(end) - at == KEY_SIZE * 2) {
            std::string run = text.substr(at, KEY_SIZE * 2);
            std::ranges::transform(run, run.begin(), [](char c) { return std::toupper(c); });
            found.insert(std::move(run));
        }
        at = end;
    }
    if (found.empty()) {
        return fmt::format("There's no trophy key in {}: it should hold the 16-byte key, or "
                           "the key as 32 hex digits.",
                           name);
    }
    if (found.size() > 1) {
        return fmt::format("{} holds more than one key, so it isn't clear which is the trophy key.",
                           name);
    }
    return FromHex(*found.begin());
}

std::string InstallTrophyKey(const fs::path& file) {
    auto read = ReadTrophyKey(file);
    if (const auto* error = std::get_if<std::string>(&read)) {
        return *error;
    }
    const auto manager = KeyManager::GetInstance();
    auto keys = manager->GetAllKeys();
    keys.TrophyKeySet.ReleaseTrophyKey = std::get<std::vector<u8>>(std::move(read));
    manager->SetAllKeys(keys);
    if (!manager->SaveToFile()) {
        return "Couldn't save the key to shadPS4's keys.json.";
    }
    return {};
}

bool HaveTrophyKey() {
    return KeyManager::GetInstance()->GetAllKeys().TrophyKeySet.ReleaseTrophyKey.size() == KEY_SIZE;
}

} // namespace Launcher::Keys
