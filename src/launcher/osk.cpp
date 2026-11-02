// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <mutex>
#include <utility>

#include "common/logging/log.h"
#include "launcher/osk.h"

namespace Launcher::Osk {

namespace {

std::atomic<u64> g_generation{0};
std::atomic_bool g_available{false};

std::mutex g_mutex;
u32 g_open = 0; // The launcher's keyboard, while it has one of ours
std::vector<Event> g_events;

// UTF-8 to code points; a malformed byte stands for itself
std::u32string Decode(std::string_view text) {
    std::u32string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<u8>(text[i]);
        const size_t n = c < 0x80           ? 1
                         : (c >> 5) == 0x6  ? 2
                         : (c >> 4) == 0xE  ? 3
                         : (c >> 3) == 0x1E ? 4
                                            : 0;
        if (n == 0 || i + n > text.size()) {
            out.push_back(c);
            i++;
            continue;
        }
        char32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (size_t k = 1; k < n; k++) {
            cp = (cp << 6) | (static_cast<u8>(text[i + k]) & 0x3F);
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}

std::string Encode(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t cp : text) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

u32 Utf16Units(char32_t cp) {
    return cp >= 0x10000 ? 2 : 1;
}

u32 Utf16Length(std::u32string_view text) {
    u32 n = 0;
    for (const char32_t cp : text) {
        n += Utf16Units(cp);
    }
    return n;
}

lc_session* Session() {
    return lc_get();
}

} // namespace

u64 Generation() {
    return g_generation.load();
}

bool Available() {
    return g_available.load();
}

u32 Open(const lc_osk_params& params) {
    lc_session* session = Session();
    if (session == nullptr || !g_available.load()) {
        return 0;
    }
    const u32 osk = lc_osk_open(session, &params);
    std::scoped_lock lock(g_mutex);
    g_open = osk;
    g_events.clear();
    if (osk != 0) {
        LOG_INFO(Lib_ImeDialog, "Launcher: the launcher's keyboard {} takes the dialog", osk);
    }
    return osk;
}

std::vector<Event> Take(u32 osk) {
    std::scoped_lock lock(g_mutex);
    if (osk == 0 || osk != g_open) {
        return {};
    }
    return std::exchange(g_events, {});
}

bool Owns(u32 osk) {
    std::scoped_lock lock(g_mutex);
    return osk != 0 && osk == g_open;
}

void Text(u32 osk, u64 seq, std::string_view text, u32 cursor) {
    lc_session* session = Session();
    if (session == nullptr || !Owns(osk)) {
        return;
    }
    const std::string t(text);
    lc_osk_text(session, osk, seq, t.c_str(), cursor);
}

void Close(u32 osk, lc_osk_result result) {
    {
        std::scoped_lock lock(g_mutex);
        if (osk == 0 || osk != g_open) {
            return;
        }
        g_open = 0;
        g_events.clear();
    }
    if (lc_session* session = Session()) {
        lc_osk_close(session, osk, result);
    }
}

void OnAvailable(bool available) {
    LOG_INFO(Lib_ImeDialog, "Launcher: the launcher's keyboard is {}",
             available ? "available" : "not available");
    {
        std::scoped_lock lock(g_mutex);
        if (!available) {
            // The library closed it; the dialog shows shadPS4's own
            g_open = 0;
            g_events.clear();
        }
    }
    g_available = available;
    g_generation++;
}

void OnEdit(u32 osk, u64 seq, const lc_osk_edit& edit) {
    std::scoped_lock lock(g_mutex);
    if (osk == 0 || osk != g_open) {
        return;
    }
    g_events.push_back(Event{
        .kind = Event::Kind::Edit,
        .seq = seq,
        .op = edit.op,
        .text = edit.text != nullptr ? edit.text : "",
        .by = edit.by,
        .cursor = edit.cursor,
    });
}

void OnFinish(u32 osk, bool accepted) {
    std::scoped_lock lock(g_mutex);
    if (osk == 0 || osk != g_open) {
        return;
    }
    g_events.push_back(Event{.kind = Event::Kind::Finish, .accepted = accepted});
}

bool ApplyEdit(std::string& text, u32& cursor, const Event& edit, u32 max_utf16, bool multiline) {
    std::u32string t = Decode(text);
    size_t at = std::min<size_t>(cursor, t.size());
    const auto clean = [&](std::string_view in) {
        std::u32string s = Decode(in);
        if (!multiline) {
            std::erase_if(s, [](char32_t c) { return c == U'\n' || c == U'\r'; });
        }
        return s;
    };
    const std::u32string before = t;
    const size_t before_at = at;

    switch (edit.op) {
    case LC_OSK_INSERT: {
        std::u32string insert = clean(edit.text);
        if (max_utf16 != 0) {
            // As much as fits
            u32 room = max_utf16 > Utf16Length(t) ? max_utf16 - Utf16Length(t) : 0;
            size_t n = 0;
            while (n < insert.size() && Utf16Units(insert[n]) <= room) {
                room -= Utf16Units(insert[n]);
                n++;
            }
            insert.resize(n);
        }
        t.insert(at, insert);
        at += insert.size();
        break;
    }
    case LC_OSK_BACKSPACE:
        if (at > 0) {
            t.erase(at - 1, 1);
            at--;
        }
        break;
    case LC_OSK_DELETE:
        if (at < t.size()) {
            t.erase(at, 1);
        }
        break;
    case LC_OSK_MOVE: {
        const auto moved = static_cast<s64>(at) + edit.by;
        at = static_cast<size_t>(std::clamp<s64>(moved, 0, static_cast<s64>(t.size())));
        break;
    }
    case LC_OSK_REPLACE: {
        t = clean(edit.text);
        if (max_utf16 != 0) {
            u32 units = 0;
            size_t n = 0;
            while (n < t.size() && units + Utf16Units(t[n]) <= max_utf16) {
                units += Utf16Units(t[n]);
                n++;
            }
            t.resize(n);
        }
        at = std::min<size_t>(edit.cursor, t.size());
        break;
    }
    default:
        return false;
    }

    if (t == before && at == before_at) {
        return false;
    }
    text = Encode(t);
    cursor = static_cast<u32>(at);
    return true;
}

u32 CharactersAtUtf16(std::string_view text, u32 utf16) {
    const std::u32string t = Decode(text);
    u32 units = 0;
    u32 n = 0;
    for (const char32_t cp : t) {
        if (units + Utf16Units(cp) > utf16) {
            break;
        }
        units += Utf16Units(cp);
        n++;
    }
    return n;
}

u32 Utf16AtCharacter(std::string_view text, u32 characters) {
    const std::u32string t = Decode(text);
    return Utf16Length(std::u32string_view(t).substr(0, std::min<size_t>(characters, t.size())));
}

} // namespace Launcher::Osk
