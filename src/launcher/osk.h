// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <launcher_client.h>

#include "common/types.h"

// The game launcher's on-screen keyboard: while the launcher can draw keyboards into the game's
// frames (osk_available), the IME dialog a game opens is drawn by the launcher instead of
// shadPS4's own. The dialog keeps the text, so its rules (the length, the game's filter) apply:
// the launcher's keys are edits to it, and the text goes back after each change. Outside launcher
// mode, or while the launcher can't, nothing changes.
//
// The launcher's callbacks queue what it sends; the dialog takes it on the present thread.
namespace Launcher::Osk {

// Changes whenever the launcher starts or stops being able to draw keyboards
[[nodiscard]] u64 Generation();
[[nodiscard]] bool Available();

// Asks the launcher for a keyboard: its number, or 0 to show shadPS4's own.
[[nodiscard]] u32 Open(const lc_osk_params& params);

struct Event {
    enum class Kind { Edit, Finish };
    Kind kind = Kind::Edit;
    u64 seq = 0;
    lc_osk_op op = LC_OSK_INSERT;
    std::string text;
    s32 by = 0;
    u32 cursor = 0;
    bool accepted = false;
};

// What the launcher sent for keyboard `osk` since the last call
[[nodiscard]] std::vector<Event> Take(u32 osk);

// Whether keyboard `osk` is still the launcher's: false once it can't draw keyboards any more,
// which closes it as far as the launcher is concerned (show shadPS4's own, with the text as it
// is).
[[nodiscard]] bool Owns(u32 osk);

// The text after a change, with the last edit it includes; `cursor` in characters
void Text(u32 osk, u64 seq, std::string_view text, u32 cursor);

// Keyboard `osk` closed
void Close(u32 osk, lc_osk_result result);

// The launcher's callbacks (session.cpp)
void OnAvailable(bool available);
void OnEdit(u32 osk, u64 seq, const lc_osk_edit& edit);
void OnFinish(u32 osk, bool accepted);

// Applies one edit to `text` (UTF-8) with its cursor in characters, keeping at most `max_utf16`
// UTF-16 units (0 for no limit) and no line breaks unless `multiline`. False if nothing changed.
bool ApplyEdit(std::string& text, u32& cursor, const Event& edit, u32 max_utf16, bool multiline);

// Positions in UTF-8 text: characters before a UTF-16 offset, and UTF-16 units before a character
[[nodiscard]] u32 CharactersAtUtf16(std::string_view text, u32 utf16);
[[nodiscard]] u32 Utf16AtCharacter(std::string_view text, u32 characters);

} // namespace Launcher::Osk
