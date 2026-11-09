// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <functional>
#include <fmt/format.h>

#include <map>
#include <mutex>
#include <SDL3/SDL_hints.h>

#include "common/ab_switch.h"
#include "common/path_util.h"
#include "core/devtools/layer.h"
#include "core/emulator_settings.h"
#include "core/libraries/audio/audioout.h"
#include "input/input_handler.h"
#include "launcher/settings.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;

namespace Launcher::Settings {

namespace {

using Options = std::vector<std::pair<std::string, std::string>>;

struct Entry {
    const char* key;
    const char* label;
    const char* section;
    const char* help;
    Row::Kind kind;
    Options options;
    std::function<std::string()> get;
    // False if the value isn't one it takes
    std::function<bool(const std::string&)> set;
    bool runner = false;
    // A choice that also takes values that aren't among its options (sizes, steps), as shadPS4's
    // own config may hold one
    bool free = false;
    // Makes a change take effect while the game runs; none for a setting read only at boot
    std::function<void()> live;
};

// A setting that can change while the game runs, with what makes the change take effect
Entry Live(Entry entry, std::function<void()> refresh) {
    entry.live = std::move(refresh);
    return entry;
}

// Read where it's used, so a change takes effect by itself
void Nothing() {}

// The window follows its size or fullscreen setting on the main thread (WindowSDL)
void RefreshWindow(s32 changed) {
    SDL_Event event{};
    event.type = SDL_EVENT_LAUNCHER_WINDOW;
    event.user.code = changed;
    SDL_PushEvent(&event);
}

void RefreshWindowSize() {
    RefreshWindow(LAUNCHER_WINDOW_SIZE);
}

void RefreshDisplayMode() {
    RefreshWindow(LAUNCHER_WINDOW_MODE);
}

void RefreshUpscaling() {
    if (presenter) {
        auto& fsr = presenter->GetFsrSettingsRef();
        fsr.enable = EmulatorSettings.IsFsrEnabled();
        fsr.use_rcas = EmulatorSettings.IsRcasEnabled();
        fsr.rcas_attenuation = static_cast<float>(EmulatorSettings.GetRcasAttenuation() / 1000.f);
    }
}

std::mutex g_loaded_mutex;
std::map<std::string, std::string, std::less<>> g_loaded;

constexpr char NEXT_BOOT[] = " Takes effect when the game next starts.";

std::optional<long> ParseNumber(std::string_view text) {
    long value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::pair<u32, u32>> ParseSize(std::string_view text) {
    const auto x = text.find('x');
    if (x == std::string_view::npos) {
        return std::nullopt;
    }
    const auto w = ParseNumber(text.substr(0, x));
    const auto h = ParseNumber(text.substr(x + 1));
    if (!w || !h || *w < 320 || *h < 240 || *w > 16384 || *h > 16384) {
        return std::nullopt;
    }
    return std::pair<u32, u32>{static_cast<u32>(*w), static_cast<u32>(*h)};
}

std::string Bool(bool value) {
    return value ? "true" : "false";
}

std::optional<bool> ParseBool(std::string_view text) {
    if (text == "true") {
        return true;
    }
    if (text == "false") {
        return false;
    }
    return std::nullopt;
}

Entry Toggle(const char* key, const char* label, const char* section, const char* help,
             std::function<bool()> get, std::function<void(bool)> set) {
    return Entry{key,
                 label,
                 section,
                 help,
                 Row::Kind::Toggle,
                 {},
                 [get] { return Bool(get()); },
                 [set](const std::string& value) {
                     const auto b = ParseBool(value);
                     if (b) {
                         set(*b);
                     }
                     return b.has_value();
                 }};
}

// A folder, picked with the launcher's picker. Empty is shadPS4's own folder for it.
Entry Folder(const char* key, const char* label, const char* section, const char* help,
             std::function<std::filesystem::path()> get,
             std::function<void(const std::filesystem::path&)> set) {
    Entry entry{key,
                label,
                section,
                help,
                Row::Kind::Folder,
                {},
                [get] { return Common::FS::PathToUTF8String(get()); },
                [set](const std::string& value) {
                    set(std::filesystem::path(std::u8string(value.begin(), value.end())));
                    return true;
                }};
    entry.free = true;
    return entry;
}

Entry Choice(const char* key, const char* label, const char* section, const char* help,
             Options options, std::function<std::string()> get,
             std::function<void(const std::string&)> set) {
    return Entry{key,
                 label,
                 section,
                 help,
                 Row::Kind::Choice,
                 std::move(options),
                 std::move(get),
                 [set](const std::string& value) {
                     set(value);
                     return true;
                 }};
}

Entry Size(const char* key, const char* label, const char* section, const char* help,
           std::initializer_list<const char*> sizes, std::function<std::pair<u32, u32>()> get,
           std::function<void(u32, u32)> set) {
    Options options;
    for (const auto* size : sizes) {
        options.emplace_back(size, size);
    }
    return Entry{key,
                 label,
                 section,
                 help,
                 Row::Kind::Choice,
                 std::move(options),
                 [get] {
                     const auto [w, h] = get();
                     return fmt::format("{}x{}", w, h);
                 },
                 [set](const std::string& value) {
                     const auto size = ParseSize(value);
                     if (size) {
                         set(size->first, size->second);
                     }
                     return size.has_value();
                 },
                 false,
                 true};
}

// A number offered as steps, as the launcher's game settings can't be free numbers
Entry Steps(const char* key, const char* label, const char* section, const char* help,
            std::vector<long> steps, std::function<std::string(long)> step_label,
            std::function<long()> get, std::function<void(long)> set) {
    Options options;
    for (const auto step : steps) {
        options.emplace_back(std::to_string(step), step_label(step));
    }
    return Entry{key,
                 label,
                 section,
                 help,
                 Row::Kind::Choice,
                 std::move(options),
                 [get] { return std::to_string(get()); },
                 [set](const std::string& value) {
                     const auto number = ParseNumber(value);
                     if (number) {
                         set(*number);
                     }
                     return number.has_value();
                 },
                 false,
                 true};
}

constexpr std::array<const char*, 31> LANGUAGES = {
    "Japanese",
    "English (US)",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Dutch",
    "Portuguese (Portugal)",
    "Russian",
    "Korean",
    "Chinese (Traditional)",
    "Chinese (Simplified)",
    "Finnish",
    "Swedish",
    "Danish",
    "Norwegian",
    "Polish",
    "Portuguese (Brazil)",
    "English (UK)",
    "Turkish",
    "Spanish (Latin America)",
    "Arabic",
    "French (Canada)",
    "Czech",
    "Hungarian",
    "Greek",
    "Romanian",
    "Thai",
    "Vietnamese",
    "Indonesian",
    "Ukrainian",
};

const std::vector<Entry>& Entries() {
    // Applied as game-specific values, which win over both shadPS4's global and per-game config
    constexpr bool G = true;
    static const std::vector<Entry> entries = [] {
        std::vector<Entry> e;

        // Display
        e.push_back(Live(Size(
                             "window-size", "Window size", "Display",
                             "The game window's size when it isn't fullscreen.",
                             {"1280x720", "1600x900", "1920x1080", "2560x1440", "3840x2160"},
                             [] {
                                 return std::pair{EmulatorSettings.GetWindowWidth(),
                                                  EmulatorSettings.GetWindowHeight()};
                             },
                             [](u32 w, u32 h) {
                                 EmulatorSettings.SetWindowWidth(w, G);
                                 EmulatorSettings.SetWindowHeight(h, G);
                             }),
                         RefreshWindowSize));
        e.push_back(Live(
            Choice(
                "display-mode", "Display mode", "Display",
                "Exclusive fullscreen changes the display's mode; borderless covers the desktop.",
                {{"windowed", "Windowed"},
                 {"fullscreen", "Fullscreen"},
                 {"borderless", "Borderless fullscreen"}},
                [] {
                    if (!EmulatorSettings.IsFullScreen()) {
                        return std::string("windowed");
                    }
                    return std::string(EmulatorSettings.GetFullScreenMode() == "Fullscreen"
                                           ? "fullscreen"
                                           : "borderless");
                },
                [](const std::string& value) {
                    EmulatorSettings.SetFullScreen(value != "windowed", G);
                    if (value != "windowed") {
                        EmulatorSettings.SetFullScreenMode(
                            value == "fullscreen" ? "Fullscreen" : "Fullscreen (Borderless)", G);
                    }
                }),
            RefreshDisplayMode));
        e.push_back(Size(
            "output-resolution", "Output resolution", "Display",
            "The resolution games are told the console outputs.",
            {"1280x720", "1920x1080", "2560x1440", "3840x2160"},
            [] {
                return std::pair{EmulatorSettings.GetInternalScreenWidth(),
                                 EmulatorSettings.GetInternalScreenHeight()};
            },
            [](u32 w, u32 h) {
                EmulatorSettings.SetInternalScreenWidth(w, G);
                EmulatorSettings.SetInternalScreenHeight(h, G);
            }));
        e.push_back(Live(
            Choice(
                "present-mode", "Present mode", "Display",
                "Fifo waits for the display, Mailbox replaces waiting frames, Immediate can tear.",
                {{"Mailbox", "Mailbox"}, {"Fifo", "Fifo (vsync)"}, {"Immediate", "Immediate"}},
                [] { return EmulatorSettings.GetPresentMode(); },
                [](const std::string& value) { EmulatorSettings.SetPresentMode(value, G); }),
            [] {
                if (presenter) {
                    presenter->RecreateSwapchain();
                }
            }));
        e.push_back(Toggle(
            "hdr", "HDR", "Display", "Output HDR10 when the game and the display support it.",
            [] { return EmulatorSettings.IsHdrAllowed(); },
            [](bool v) { EmulatorSettings.SetHdrAllowed(v, G); }));
        e.push_back(
            Live(Toggle(
                     "fsr", "FSR upscaling", "Display",
                     "Upscale the game's frame to the window with AMD FidelityFX Super Resolution.",
                     [] { return EmulatorSettings.IsFsrEnabled(); },
                     [](bool v) { EmulatorSettings.SetFsrEnabled(v, G); }),
                 RefreshUpscaling));
        e.push_back(
            Live(Toggle(
                     "rcas", "RCAS sharpening", "Display", "Sharpen the upscaled frame (with FSR).",
                     [] { return EmulatorSettings.IsRcasEnabled(); },
                     [](bool v) { EmulatorSettings.SetRcasEnabled(v, G); }),
                 RefreshUpscaling));
        e.push_back(Live(Steps(
                             "rcas-attenuation", "Sharpening strength", "Display",
                             "How much RCAS holds back: lower is sharper.",
                             {0, 100, 250, 500, 750, 1000, 1500, 2000},
                             [](long v) { return fmt::format("{:.2f}", v / 1000.0); },
                             [] { return EmulatorSettings.GetRcasAttenuation(); },
                             [](long v) {
                                 EmulatorSettings.SetRcasAttenuation(
                                     static_cast<int>(std::clamp(v, 0L, 3000L)), G);
                             }),
                         RefreshUpscaling));
        e.push_back(Steps(
            "vblank-frequency", "Vblank frequency", "Display",
            "The emulated display's refresh rate. Above 60 Hz speeds up games that count vblanks.",
            {60, 120, 144, 240}, [](long v) { return fmt::format("{} Hz", v); },
            [] { return static_cast<long>(EmulatorSettings.GetVblankFrequency()); },
            [](long v) {
                EmulatorSettings.SetVblankFrequency(static_cast<u32>(std::clamp(v, 30L, 480L)), G);
            }));
        e.push_back(Toggle(
            "splash", "Splash screen", "Display", "Show the game's splash screen while it starts.",
            [] { return EmulatorSettings.IsShowSplash(); },
            [](bool v) { EmulatorSettings.SetShowSplash(v, G); }));
        e.push_back(
            Live(Toggle(
                     "fps-counter", "Frame rate", "Display", "Show shadPS4's frame rate counter.",
                     [] { return EmulatorSettings.IsShowFpsCounter(); },
                     [](bool v) { EmulatorSettings.SetShowFpsCounter(v, G); }),
                 [] { Overlay::SetSimpleFps(EmulatorSettings.IsShowFpsCounter()); }));
        e.push_back(
            Live(Choice(
                     "hide-cursor", "Hide the cursor", "Display", "When the mouse cursor hides.",
                     {{"0", "Never"}, {"1", "When idle"}, {"2", "Always"}},
                     [] { return std::to_string(EmulatorSettings.GetCursorState()); },
                     [](const std::string& value) {
                         EmulatorSettings.SetCursorState(std::stoi(value), G);
                     }),
                 Nothing));

        // Accuracy
        e.push_back(Choice(
            "readbacks", "GPU readbacks", "Accuracy",
            "Copy what the GPU writes back to the game's memory. Some games need it; it's slower.",
            {{"0", "Off"}, {"1", "Relaxed"}, {"2", "Precise"}, {"3", "Fenced"}},
            [] { return std::to_string(EmulatorSettings.GetReadbacksMode()); },
            [](const std::string& value) {
                EmulatorSettings.SetReadbacksMode(static_cast<u32>(std::stoi(value)), G);
            }));
        e.push_back(Toggle(
            "fault-load-emulation", "Finish reads without readbacks", "Accuracy",
            "With Precise or Fenced readbacks, finish a game read that touches none of the GPU's "
            "data from memory instead of waiting for the GPU. Experimental; off by default.",
            [] { return EmulatorSettings.IsFaultLoadEmulation(); },
            [](bool v) { EmulatorSettings.SetFaultLoadEmulation(v, G); }));
        // A/B testing: flip between two variants of an experiment while the game runs
        // (e.g. Precise vs Fenced readbacks), which compares them with far less noise than
        // separate runs. The active variant shows under the FPS counter.
        e.push_back(Live(Toggle(
                             "ab-variant-b", "A/B test: second variant", "Accuracy",
                             "Switch the running experiment to its second variant "
                             "(Fenced, when readbacks are Precise or Fenced).",
                             [] { return Common::AbSwitch::IsB(); },
                             [](bool v) { Common::AbSwitch::Set(v); }),
                         Nothing));
        e.push_back(Live(Choice(
                             "ab-auto-switch", "A/B test: auto switch", "Accuracy",
                             "Flip the A/B variant on a timer, for unattended comparisons.",
                             {{"0", "Off"}, {"10", "Every 10 s"}, {"30", "Every 30 s"}},
                             [] { return std::to_string(EmulatorSettings.GetAbSwitchSeconds()); },
                             [](const std::string& value) {
                                 EmulatorSettings.SetAbSwitchSeconds(
                                     static_cast<u32>(std::stoi(value)), G);
                             }),
                         Nothing));
        e.push_back(Toggle(
            "readback-linear-images", "Read back linear images", "Accuracy",
            "Also copy linear images the GPU writes back to memory.",
            [] { return EmulatorSettings.IsReadbackLinearImagesEnabled(); },
            [](bool v) { EmulatorSettings.SetReadbackLinearImagesEnabled(v, G); }));
        e.push_back(Toggle(
            "direct-memory-access", "Direct memory access", "Accuracy",
            "Let shaders reach the game's memory directly, as some games expect.",
            [] { return EmulatorSettings.IsDirectMemoryAccessEnabled(); },
            [](bool v) { EmulatorSettings.SetDirectMemoryAccessEnabled(v, G); }));
        e.push_back(Toggle(
            "copy-gpu-buffers", "Copy GPU buffers", "Accuracy",
            "Copy the game's command buffers before running them, for games that rewrite them.",
            [] { return EmulatorSettings.IsCopyGpuBuffers(); },
            [](bool v) { EmulatorSettings.SetCopyGpuBuffers(v, G); }));
        e.push_back(Toggle(
            "pipeline-cache", "Pipeline cache", "Accuracy",
            "Keep compiled pipelines on disk, so later runs stutter less.",
            [] { return EmulatorSettings.IsPipelineCacheEnabled(); },
            [](bool v) { EmulatorSettings.SetPipelineCacheEnabled(v, G); }));
        // Read for each new pipeline, so a change applies to the next one
        e.push_back(
            Live(Choice(
                     "async-pipelines", "Shader compiling", "Accuracy",
                     "How new shaders get compiled. Pausing the game stutters. In the background, "
                     "what a shader draws is missing for a moment if the game draws it every "
                     "frame; skip all also skips one-off draws, which can stay wrong. Quick "
                     "first compiles a slower version at once, then swaps in the full one.",
                     {{"0", "Pause the game"},
                      {"1", "In the background"},
                      {"2", "Quick first"},
                      {"3", "In the background, skip all"}},
                     [] { return std::to_string(EmulatorSettings.GetAsyncPipelines()); },
                     [](const std::string& value) {
                         EmulatorSettings.SetAsyncPipelines(static_cast<u32>(std::stoi(value)), G);
                     }),
                 Nothing));
        e.push_back(
            Live(Choice(
                     "async-pipelines-ab", "Shader compiling: A/B test", "Accuracy",
                     "Switch shader compiling between pausing and the mode above on a timer, to "
                     "compare them in one run. The log says which is on.",
                     {{"0", "Off"}, {"10", "Every 10 s"}, {"30", "Every 30 s"}},
                     [] { return std::to_string(EmulatorSettings.GetAsyncPipelinesAbSeconds()); },
                     [](const std::string& value) {
                         EmulatorSettings.SetAsyncPipelinesAbSeconds(
                             static_cast<u32>(std::stoi(value)), G);
                     }),
                 Nothing));
        e.push_back(Toggle(
            "descriptor-heap", "Descriptor heap", "Accuracy",
            "Give shaders their textures and buffers through the graphics driver's descriptor "
            "heap, "
            "where it has one. Experimental: it's the groundwork for compiling shaders ahead of "
            "time.",
            [] { return EmulatorSettings.IsDescriptorHeap(); },
            [](bool v) { EmulatorSettings.SetDescriptorHeap(v, G); }));
        e.push_back(Toggle(
            "shader-objects", "Shader objects", "Accuracy",
            "Compile each shader once and reuse it for every draw, instead of compiling a "
            "pipeline for each combination. Experimental, and needs the descriptor heap.",
            [] { return EmulatorSettings.IsShaderObjects(); },
            [](bool v) { EmulatorSettings.SetShaderObjects(v, G); }));
        e.push_back(Toggle(
            "compile-lookahead", "Compile ahead", "Accuracy",
            "When a draw needs new shaders, read ahead through the frame's commands and start "
            "compiling the shaders of later draws too. Experimental, and only with a "
            "background shader compiling mode.",
            [] { return EmulatorSettings.IsCompileLookahead(); },
            [](bool v) { EmulatorSettings.SetCompileLookahead(v, G); }));
        e.push_back(Toggle(
            "shader-prescan", "Find shaders early", "Accuracy",
            "Look for the game's shaders in memory as it loads them, before their first draw, "
            "and remember them for the next launch. Experimental, Linux only.",
            [] { return EmulatorSettings.IsShaderPrescan(); },
            [](bool v) { EmulatorSettings.SetShaderPrescan(v, G); }));
        e.push_back(Toggle(
            "fewer-shader-variants", "Fewer shader variants", "Accuracy",
            "Compile one shader for draws that differ only in ways that don't change it, such as "
            "the exact format they draw to. With the descriptor heap, shaders also read buffer "
            "formats as they run. Experimental.",
            [] { return EmulatorSettings.IsFewerShaderVariants(); },
            [](bool v) { EmulatorSettings.SetFewerShaderVariants(v, G); }));
        e.push_back(Choice(
            "shader-precompile-threads", "Compile found shaders early", "Accuracy",
            "Compile the shaders \"Find shaders early\" finds before the game first draws with "
            "them, on this many threads. More threads catch up sooner but take more CPU from the "
            "game. Needs the descriptor heap. Experimental.",
            {{"0", "Off"}, {"1", "1 thread"}, {"2", "2 threads"}, {"4", "4 threads"}},
            [] { return std::to_string(EmulatorSettings.GetShaderPrecompileThreads()); },
            [](const std::string& value) {
                EmulatorSettings.SetShaderPrecompileThreads(static_cast<u32>(std::stoi(value)), G);
            }));

        // System
        e.push_back(Toggle(
            "neo", "PS4 Pro mode", "System", "Tell games they run on a PS4 Pro.",
            [] { return EmulatorSettings.IsNeo(); },
            [](bool v) { EmulatorSettings.SetNeo(v, G); }));
        e.push_back(Toggle(
            "network", "Connected to the network", "System", "Tell games the console is online.",
            [] { return EmulatorSettings.IsConnectedToNetwork(); },
            [](bool v) { EmulatorSettings.SetConnectedToNetwork(v, G); }));
        {
            Options languages;
            for (size_t i = 0; i < LANGUAGES.size(); i++) {
                languages.emplace_back(std::to_string(i), LANGUAGES[i]);
            }
            auto language = Choice(
                "console-language", "Console language", "Profile",
                "The console's language, which games and trophy lists use.", std::move(languages),
                [] { return std::to_string(EmulatorSettings.GetConsoleLanguage()); },
                [](const std::string& value) {
                    EmulatorSettings.SetConsoleLanguage(std::stoi(value), G);
                });
            language.runner = true;
            e.push_back(std::move(language));
        }

        // What the player dumped from their own console, for games that need it
        auto modules = Folder(
            "sys-modules-dir", "System modules", "System files",
            "A folder of system modules (.sprx files) from your PS4's firmware, for games that "
            "need one shadPS4 doesn't emulate. Modules in a folder named after a game's title id "
            "are for that game alone.",
            [] { return EmulatorSettings.GetSysModulesDir(); },
            [](const std::filesystem::path& dir) { EmulatorSettings.SetSysModulesDir(dir); });
        modules.runner = true;
        e.push_back(std::move(modules));
        auto fonts = Folder(
            "fonts-dir", "System fonts", "System files",
            "A folder holding the font and font2 folders from your PS4's system software, for "
            "games that draw text with the console's fonts.",
            [] { return EmulatorSettings.GetFontsDir(); },
            [](const std::filesystem::path& dir) { EmulatorSettings.SetFontsDir(dir); });
        fonts.runner = true;
        e.push_back(std::move(fonts));

        // Audio
        e.push_back(Live(Steps(
                             "volume", "Volume", "Audio", "The game's volume.",
                             {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150},
                             [](long v) { return fmt::format("{}%", v); },
                             [] { return static_cast<long>(EmulatorSettings.GetVolumeSlider()); },
                             [](long v) {
                                 EmulatorSettings.SetVolumeSlider(
                                     static_cast<int>(std::clamp(v, 0L, 500L)), G);
                             }),
                         Libraries::AudioOut::AdjustVol));
        e.push_back(Choice(
            "audio-backend", "Audio backend", "Audio", "How sound reaches the system.",
            {{"0", "SDL"}, {"1", "OpenAL"}},
            [] { return std::to_string(EmulatorSettings.GetAudioBackend()); },
            [](const std::string& value) {
                EmulatorSettings.SetAudioBackend(static_cast<u32>(std::stoi(value)), G);
            }));

        // Controls
        e.push_back(Toggle(
            "circle-enter", "Confirm with Circle", "Controls",
            "Japanese-style buttons: Circle confirms and Cross cancels, in games that ask.",
            [] { return EmulatorSettings.IsCircleEnter(); },
            [](bool v) { EmulatorSettings.SetCircleEnter(v, G); }));
        e.push_back(Toggle(
            "motion-controls", "Motion controls", "Controls",
            "Pass the controller's gyro and accelerometer to games.",
            [] { return EmulatorSettings.IsMotionControlsEnabled(); },
            [](bool v) { EmulatorSettings.SetMotionControlsEnabled(v, G); }));
        e.push_back(Live(Toggle(
                             "background-input", "Input in the background", "Controls",
                             "Keep reading the controller while the game window isn't focused.",
                             [] { return EmulatorSettings.IsBackgroundControllerInput(); },
                             [](bool v) { EmulatorSettings.SetBackgroundControllerInput(v, G); }),
                         [] {
                             SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS,
                                         EmulatorSettings.IsBackgroundControllerInput() ? "1"
                                                                                        : "0");
                         }));
        return e;
    }();
    return entries;
}

} // namespace

std::vector<Row> Rows() {
    std::vector<Row> rows;
    for (const auto& e : Entries()) {
        Row row;
        row.key = e.key;
        row.label = e.label;
        row.section = e.section;
        row.help = e.help;
        if (!e.live) {
            row.help += NEXT_BOOT;
        }
        row.kind = e.kind;
        row.options = e.options;
        row.default_value = e.get();
        row.runner = e.runner;
        row.live = static_cast<bool>(e.live);
        if (row.kind == Row::Kind::Choice && std::ranges::none_of(row.options, [&](const auto& o) {
                return o.first == row.default_value;
            })) {
            // A value set in shadPS4's own config that isn't one of the steps
            row.options.emplace_back(row.default_value, row.default_value);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void CaptureLoaded() {
    std::scoped_lock lock(g_loaded_mutex);
    g_loaded.clear();
    for (const auto& e : Entries()) {
        g_loaded.emplace(e.key, e.get());
    }
}

std::string Apply(std::string_view key, const std::optional<std::string>& value) {
    const auto& entries = Entries();
    const auto it = std::ranges::find_if(entries, [&](const Entry& e) { return e.key == key; });
    if (it == entries.end()) {
        return fmt::format("shadPS4 has no setting {}", key);
    }
    if (!value) {
        // Back to what the loaded config says
        std::string loaded;
        {
            std::scoped_lock lock(g_loaded_mutex);
            const auto l = g_loaded.find(key);
            if (l == g_loaded.end()) {
                return {};
            }
            loaded = l->second;
        }
        it->set(loaded);
        return {};
    }
    const bool listed =
        it->kind == Row::Kind::Toggle ||
        std::ranges::any_of(it->options, [&](const auto& o) { return o.first == *value; });
    if ((!listed && !it->free) || !it->set(*value)) {
        return fmt::format("{} can't be {}", it->label, *value);
    }
    return {};
}

std::string ApplyLive(std::string_view key, const std::optional<std::string>& value) {
    if (auto error = Apply(key, value); !error.empty()) {
        return error;
    }
    const auto& entries = Entries();
    const auto it = std::ranges::find_if(entries, [&](const Entry& e) { return e.key == key; });
    if (!it->live) {
        return fmt::format("{} takes effect when the game next starts.", it->label);
    }
    it->live();
    return {};
}

} // namespace Launcher::Settings
