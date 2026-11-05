// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/file_format/npbind.h"

namespace {

void PutBe(std::vector<u8>& out, u64 value, int bytes) {
    for (int i = bytes - 1; i >= 0; i--) {
        out.push_back(static_cast<u8>(value >> (i * 8)));
    }
}

void PutEntry(std::vector<u8>& out, u16 type, const std::string& text, u16 size) {
    PutBe(out, type, 2);
    PutBe(out, size, 2);
    std::vector<u8> data(size, 0);
    std::memcpy(data.data(), text.data(), std::min<size_t>(text.size(), size));
    out.insert(out.end(), data.begin(), data.end());
}

// An npbind.dat with one body per (npcommid, trophy field)
std::vector<u8> Npbind(const std::vector<std::pair<std::string, std::string>>& sets) {
    std::vector<u8> out;
    PutBe(out, NPBIND_MAGIC, 4);
    PutBe(out, 1, 4);
    PutBe(out, 0, 8);
    PutBe(out, 0x180, 8);
    PutBe(out, sets.size(), 8);
    out.resize(sizeof(NpBindHeader), 0);
    for (const auto& [id, trophy] : sets) {
        PutEntry(out, 0x10, id, 12);
        PutEntry(out, 0x11, trophy, 12);
        PutEntry(out, 0x12, "", 176);
        PutEntry(out, 0x13, "", 16);
        out.resize(out.size() + 0x98, 0);
    }
    out.resize(out.size() + 20, 0);
    return out;
}

TEST(NpBindTest, TrophyNumbersFollowTheTrophyField) {
    // Uncharted: The Nathan Drake Collection: three sets in trophy01..03.trp
    const auto data = Npbind({{"NPWR09798_00", "1"}, {"NPWR09799_00", "2"}, {"NPWR09801_00", "3"}});
    NPBindFile npbind;
    ASSERT_TRUE(npbind.Load(std::span<const u8>(data)));
    EXPECT_EQ(npbind.GetNpCommIds(),
              (std::vector<std::string>{"NPWR09798_00", "NPWR09799_00", "NPWR09801_00"}));
    EXPECT_EQ(npbind.GetTrophyNumbers(), (std::vector<s32>{1, 2, 3}));
}

TEST(NpBindTest, TrophyNumbersFallBackToPosition) {
    const auto data = Npbind({{"NPWR00001_00", "0"}, {"NPWR00002_00", ""}, {"NPWR00003_00", "x"}});
    NPBindFile npbind;
    ASSERT_TRUE(npbind.Load(std::span<const u8>(data)));
    EXPECT_EQ(npbind.GetTrophyNumbers(), (std::vector<s32>{0, 1, 2}));
}

} // namespace
