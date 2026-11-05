// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cctype>

#include "common/key_manager.h"
#include "launcher/keys.h"

namespace Launcher::Keys {

namespace {

constexpr size_t KEY_SIZE = 16;

int HexValue(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower >= 'a' && lower <= 'f' ? lower - 'a' + 10 : -1;
}

} // namespace

std::variant<std::vector<u8>, std::string> ParseTrophyKey(std::string_view text) {
    std::string digits;
    for (const char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (HexValue(c) < 0) {
            return std::string("A trophy key is hex digits (0-9 and A-F) only.");
        }
        digits.push_back(c);
    }
    if (digits.size() != KEY_SIZE * 2) {
        return "A trophy key is 32 hex digits; that's " + std::to_string(digits.size()) + ".";
    }
    std::vector<u8> key;
    for (size_t i = 0; i < digits.size(); i += 2) {
        key.push_back(static_cast<u8>(HexValue(digits[i]) << 4 | HexValue(digits[i + 1])));
    }
    return key;
}

std::vector<u8> TrophyKey() {
    auto key = KeyManager::GetInstance()->GetAllKeys().TrophyKeySet.ReleaseTrophyKey;
    return key.size() == KEY_SIZE ? key : std::vector<u8>{};
}

std::string TrophyKeyHex() {
    const auto key = TrophyKey();
    return key.empty() ? std::string{} : KeyManager::BytesToHexString(key);
}

std::string SaveTrophyKey(const std::vector<u8>& key) {
    const auto manager = KeyManager::GetInstance();
    auto keys = manager->GetAllKeys();
    keys.TrophyKeySet.ReleaseTrophyKey = key;
    manager->SetAllKeys(keys);
    if (!manager->SaveToFile()) {
        return "Couldn't save the key to shadPS4's keys.json.";
    }
    return {};
}

bool HaveTrophyKey() {
    return !TrophyKey().empty();
}

} // namespace Launcher::Keys
