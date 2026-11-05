// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>
#include <launcher_client.h>
#include <launcher_overlay.h>

#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "common/singleton.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/user_settings.h"
#include "input/controller.h"
#include "launcher/osk.h"
#include "launcher/session.h"
#include "launcher/settings.h"
#include "launcher/trophies.h"
#include "sdl_window.h"

extern Frontend::WindowSDL* g_window;

namespace Launcher::Session {

namespace {

// launcherd sends configure right after the welcome; this is only for a launcher that never does
constexpr u32 CONFIGURE_WAIT_MS = 5000;
// What Booting says, until the first frame
constexpr const char* BOOT_ACTIVITY = "boot";

lc_session* g_session = nullptr;
// The launcher's values, by key: configure's, the plugin's own, and the live changes since
std::mutex g_values_mutex;
std::vector<std::pair<std::string, std::optional<std::string>>> g_values;
// ApplySettings has run, so changes take effect as they come
bool g_applied = false;
std::atomic_flag g_ready;
std::atomic_flag g_achievements_sent;
// The quick menu paused the game, which it resumes when it closes. A game the player had paused
// already (shadPS4's pause key) stays paused.
std::atomic_bool g_paused_for_menu;

// While the guest is paused, one of its threads may be stopped inside the logger, holding its
// lock, so the launcher's lines from then wait here until the guest resumes. A game thread can
// hold g_held_mutex too (a trophy unlock logs through the client), so while paused it's only
// tried, never waited for.
enum class Level { Info, Warning, Error };
std::mutex g_held_mutex;
std::vector<std::pair<Level, std::string>> g_held;
std::atomic_bool g_has_held;
size_t g_dropped = 0;
constexpr size_t HELD_LINES = 256;

void Write(Level level, const std::string& line) {
    switch (level) {
    case Level::Info:
        LOG_INFO(Frontend, "Launcher: {}", line);
        break;
    case Level::Warning:
        LOG_WARNING(Frontend, "Launcher: {}", line);
        break;
    case Level::Error:
        LOG_ERROR(Frontend, "Launcher: {}", line);
        break;
    }
}

// Logs the lines held while the guest was paused. After it resumes.
void WriteHeld() {
    if (!g_has_held.load()) {
        return;
    }
    std::vector<std::pair<Level, std::string>> held;
    size_t dropped = 0;
    {
        std::scoped_lock lock(g_held_mutex);
        held.swap(g_held);
        std::swap(dropped, g_dropped);
        g_has_held = false;
    }
    for (const auto& [level, line] : held) {
        Write(level, line);
    }
    if (dropped != 0) {
        Write(Level::Warning,
              fmt::format("{} lines from while the game was paused dropped", dropped));
    }
}

void Say(Level level, std::string line) {
    if (DebugState.IsGuestThreadsPaused()) {
        std::unique_lock lock(g_held_mutex, std::try_to_lock);
        if (!lock) {
            return;
        }
        if (g_held.size() < HELD_LINES) {
            g_held.emplace_back(level, std::move(line));
        } else {
            g_dropped++;
        }
        g_has_held = true;
        return;
    }
    WriteHeld();
    Write(level, line);
}

void Log(void*, lc_log_level level, const char* line) {
    Say(level == LC_LOG_ERROR ? Level::Error : Level::Info, line);
}

void Quit(void*) {
    // Resumed before logging: a guest thread paused inside the logger holds its lock
    if (g_paused_for_menu.exchange(false)) {
        DebugState.ResumeGuestThreads();
    }
    Say(Level::Info, "quit");
    if (g_window == nullptr) {
        // Still booting, before the window that would take the event
        std::quick_exit(0);
    }
    SDL_Event event{};
    event.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&event);
}

// The quick menu opened (pause) or closed. The present thread keeps showing the frame the game
// paused on (VideoOutDriver::PresentThread), and with the overlay drawn in the game's window the
// next of those presents is copied for the shell, as the menu's background (Launcher::Overlay)
void Pause(void*, bool opened) {
    if (opened) {
        Say(Level::Info, "quick menu opened");
    }
    auto* controllers = Common::Singleton<Input::GameControllers>::Instance();
    if (opened) {
        // The pad is the menu's before the game pauses, and the game's again after it resumes
        controllers->HoldForLauncher(true);
        if (!DebugState.IsGuestThreadsPaused()) {
            g_paused_for_menu = true;
            DebugState.PauseGuestThreads();
        }
    } else {
        if (g_paused_for_menu.exchange(false)) {
            DebugState.ResumeGuestThreads();
        }
        controllers->HoldForLauncher(false);
    }
    // Logged after the resume (Say holds lines while the guest is paused)
    if (!opened) {
        Say(Level::Info, "quick menu closed");
    }
    lc_paused(g_session, DebugState.IsGuestThreadsPaused());
}

void SetValue(const std::string& key, std::optional<std::string> value) {
    const auto it = std::ranges::find_if(g_values, [&](const auto& kv) { return kv.first == key; });
    if (it != g_values.end()) {
        it->second = std::move(value);
    } else {
        g_values.emplace_back(key, std::move(value));
    }
}

// A setting changed in the launcher while the game runs
void SettingChanged(void*, const char* key, const char* value) {
    if (key == nullptr) {
        return;
    }
    const std::optional<std::string> v =
        value != nullptr ? std::optional<std::string>(value) : std::nullopt;
    std::scoped_lock lock(g_values_mutex);
    SetValue(key, v);
    if (!g_applied) {
        return; // Still booting: ApplySettings applies it with the rest
    }
    const auto error = Settings::ApplyLive(key, v);
    if (!error.empty()) {
        Say(Level::Warning, error);
    } else {
        Say(Level::Info, fmt::format("{} = {} (live)", key, v.value_or("(default)")));
    }
}

u64 NativeWindow() {
    if (g_window == nullptr || g_window->GetSDLWindow() == nullptr) {
        return 0;
    }
    SDL_Window* const window = g_window->GetSDLWindow();
#if defined(SDL_PLATFORM_WIN32)
    return reinterpret_cast<u64>(SDL_GetPointerProperty(
        SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
#elif defined(SDL_PLATFORM_MACOS)
    return Frontend::WindowNumber(window);
#else
    // Wayland windows have no id another process can use
    return static_cast<u64>(SDL_GetNumberProperty(SDL_GetWindowProperties(window),
                                                  SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
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
    cb.pause = Pause;
    cb.setting = SettingChanged;
    cb.osk_available = [](void*, bool available) { Osk::OnAvailable(available); };
    cb.osk_edit = [](void*, uint32_t osk, uint64_t seq, const lc_osk_edit* edit) {
        if (edit != nullptr) {
            Osk::OnEdit(osk, seq, *edit);
        }
    };
    cb.osk_finish = [](void*, uint32_t osk, bool accepted) { Osk::OnFinish(osk, accepted); };

    const std::string name = std::string("shadPS4 ") + Common::g_version;
    lc_connect_info info{};
    info.name = name.c_str();
#ifdef LAUNCHER_CLIENT_VULKAN
    // The presenter draws the launcher's quick menu into the game's frames, from the shell's draw
    // lists (launcher/overlay.h), and the launcher's keyboard in place of the IME dialog's own
    // (launcher/osk.h)
    if (lc_overlay_supported()) {
        info.capabilities |= LC_CAP_COMPOSITED_OVERLAY | LC_CAP_DRAW_LISTS | LC_CAP_OSK;
    }
#endif
    std::array<char, 512> error{};
    g_session = lc_connect(&info, &cb, error.data(), error.size());
    if (g_session == nullptr) {
        LOG_ERROR(Frontend, "Launcher: can't connect: {}", error.data());
        return;
    }
    lc_booting(g_session);
    // The launcher's menu and keyboard take text input from the window (launcher/input.h), which
    // would otherwise show the system's on-screen keyboard over them: Steam's on a Steam Deck, or
    // any time no keyboard is attached. SDL_ENABLE_SCREEN_KEYBOARD in the environment still wins.
    SDL_SetHint(SDL_HINT_ENABLE_SCREEN_KEYBOARD, "0");

    const lc_kv* values = nullptr;
    size_t count = 0;
    if (lc_wait_configure(g_session, CONFIGURE_WAIT_MS, &values, &count)) {
        for (size_t i = 0; i < count; i++) {
            SetValue(values[i].key, values[i].value != nullptr
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
        SetValue(setting.substr(0, eq), setting.substr(eq + 1));
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
    std::scoped_lock lock(g_values_mutex);
    const bool again = std::exchange(g_applied, true);
    // What shadPS4's config says, which a setting set back to its default goes back to
    Settings::CaptureLoaded();
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

void Booting(const char* what) {
    if (g_session == nullptr) {
        return;
    }
    lc_activity(g_session, BOOT_ACTIVITY, what, -1.0f);
}

void FirstFrame() {
    if (g_session == nullptr || g_ready.test_and_set()) {
        return;
    }
    lc_activity_done(g_session, BOOT_ACTIVITY);
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
        a.set = t.set.c_str();
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
