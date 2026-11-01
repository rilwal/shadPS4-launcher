// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>
#include <launcher_client.h>

#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "core/emulator_settings.h"
#include "core/user_settings.h"
#include "launcher/session.h"
#include "launcher/settings.h"
#include "launcher/trophies.h"
#include "sdl_window.h"

extern Frontend::WindowSDL* g_window;

namespace Launcher::Session {

namespace {

// launcherd sends configure right after the welcome; this is only for a launcher that never does
constexpr u32 CONFIGURE_WAIT_MS = 5000;

lc_session* g_session = nullptr;
std::vector<std::pair<std::string, std::optional<std::string>>> g_values;
std::atomic_flag g_ready;
std::atomic_flag g_achievements_sent;

void Log(void*, lc_log_level level, const char* line) {
    if (level == LC_LOG_ERROR) {
        LOG_ERROR(Frontend, "Launcher: {}", line);
    } else {
        LOG_INFO(Frontend, "Launcher: {}", line);
    }
}

void Quit(void*) {
    LOG_INFO(Frontend, "Launcher: quit");
    if (g_window == nullptr) {
        // Still booting, before the window that would take the event
        std::quick_exit(0);
    }
    SDL_Event event{};
    event.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&event);
}

u64 NativeWindow() {
    if (g_window == nullptr || g_window->GetSDLWindow() == nullptr) {
        return 0;
    }
    const auto props = SDL_GetWindowProperties(g_window->GetSDLWindow());
#if defined(SDL_PLATFORM_WIN32)
    return reinterpret_cast<u64>(
        SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
#elif defined(SDL_PLATFORM_MACOS)
    return static_cast<u64>(SDL_GetNumberProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_NUMBER, 0));
#else
    // Wayland windows have no id another process can use
    return static_cast<u64>(SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
#endif
}

} // namespace

void Connect(std::vector<std::string> settings) {
    if (!lc_env_present()) {
        return;
    }
    lc_callbacks cb{};
    cb.log = Log;
    cb.quit = Quit;

    const std::string name = std::string("shadPS4 ") + Common::g_version;
    lc_connect_info info{};
    info.name = name.c_str();
    std::array<char, 512> error{};
    g_session = lc_connect(&info, &cb, error.data(), error.size());
    if (g_session == nullptr) {
        LOG_ERROR(Frontend, "Launcher: can't connect: {}", error.data());
        return;
    }
    lc_booting(g_session);

    const lc_kv* values = nullptr;
    size_t count = 0;
    if (lc_wait_configure(g_session, CONFIGURE_WAIT_MS, &values, &count)) {
        for (size_t i = 0; i < count; i++) {
            g_values.emplace_back(values[i].key, values[i].value != nullptr
                                                     ? std::optional<std::string>(values[i].value)
                                                     : std::nullopt);
        }
    } else {
        LOG_WARNING(Frontend, "Launcher: no settings came before boot");
    }
    // The plugin's own come on the command line, and win over a value configure had for them
    for (const auto& setting : settings) {
        const auto eq = setting.find('=');
        if (eq == std::string::npos) {
            LOG_ERROR(Frontend, "Launcher: {} isn't key=value", setting);
            continue;
        }
        g_values.emplace_back(setting.substr(0, eq), setting.substr(eq + 1));
    }
}

bool Active() {
    return g_session != nullptr;
}

void ApplySettings() {
    if (g_session == nullptr) {
        return;
    }
    // Again after shadPS4's own settings dialog reloads its config, which says nothing new
    static std::atomic_flag applied;
    const bool again = applied.test_and_set();
    for (const auto& [key, value] : g_values) {
        const auto error = Settings::Apply(key, value);
        if (!error.empty()) {
            LOG_ERROR(Frontend, "Launcher: {}", error);
            if (!again) {
                lc_notify(g_session, "shadPS4 skipped a setting", error.c_str(), true);
            }
        } else {
            LOG_INFO(Frontend, "Launcher: {} = {}", key, value.value_or("(default)"));
        }
    }
    EmulatorSettings.SetTrophyPopupDisabled(true, true);
}

void FirstFrame() {
    if (g_session == nullptr || g_ready.test_and_set()) {
        return;
    }
    lc_ready(g_session, 0, NativeWindow());
}

void Failed(std::string_view reason, const char* code) {
    if (g_session == nullptr) {
        return;
    }
    const std::string text(reason);
    lc_failed(g_session, text.c_str(), code);
}

void TrophiesOpened(const std::map<s32, std::string>& sets, s32 user_id) {
    if (g_session == nullptr || user_id != UserManagement.GetDefaultUser().user_id ||
        g_achievements_sent.test_and_set()) {
        return;
    }
    const auto trophies =
        Trophies::List(sets, user_id, static_cast<int>(EmulatorSettings.GetConsoleLanguage()));
    std::vector<lc_achievement> list;
    list.reserve(trophies.size());
    for (const auto& t : trophies) {
        lc_achievement a{};
        a.id = t.id.c_str();
        a.name = t.name.c_str();
        a.description = t.description.c_str();
        a.hidden = t.hidden;
        a.dlc = t.dlc;
        a.grade = t.grade.empty() ? nullptr : t.grade.c_str();
        a.unlocked = t.unlocked_at.has_value();
        a.unlocked_at = t.unlocked_at.value_or(0);
        list.push_back(a);
    }
    lc_achievements(g_session, list.data(), list.size());
}

void TrophyUnlocked(s32 user_id, s32 service_label, s32 trophy_id, u64 at) {
    if (g_session == nullptr || user_id != UserManagement.GetDefaultUser().user_id) {
        return;
    }
    lc_unlocked(g_session, Trophies::Id(service_label, trophy_id).c_str(),
                at != 0 ? static_cast<int64_t>(at) : LC_UNLOCKED_NOW);
}

void Close() {
    if (g_session != nullptr) {
        lc_flush(g_session, 1000);
    }
}

} // namespace Launcher::Session
