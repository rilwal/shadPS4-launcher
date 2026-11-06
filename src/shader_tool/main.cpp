// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Runs the shader recompiler on a game's saved shaders without the emulator: the shader list
// and draw states the shader scanner keeps in the cache folder, translated for the host the
// emulator last ran on. For timing translation and scoring guesses against what draws really
// used, at the cost of a rebuild of the recompiler instead of a run of the game.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tsl/robin_map.h>
#include <xxhash.h>

#include "common/hash.h"
#include "common/soft_assert.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/shader_guess.h"
#include "video_core/shader_list.h"

namespace {

using Shader::HwStage;
using Shader::SwStage;
using namespace Vulkan;
namespace Files = VideoCore::ShaderFiles;

constexpr std::array<u32, Shader::ShaderParams::NumShaderUserData> NoUserData{};

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const char* StageName(HwStage stage) {
    switch (stage) {
    case HwStage::Fragment:
        return "fs";
    case HwStage::Vertex:
        return "vs";
    case HwStage::Geometry:
        return "gs";
    case HwStage::Export:
        return "es";
    case HwStage::Hull:
        return "hs";
    case HwStage::Local:
        return "ls";
    case HwStage::Compute:
        return "cs";
    default:
        return "?";
    }
}

SwStage SwStageOf(HwStage stage) {
    return stage == HwStage::Compute  ? SwStage::Compute
           : stage == HwStage::Vertex ? SwStage::Vertex
                                      : SwStage::Fragment;
}

struct Game {
    Files::ShaderList list;
    Files::ShaderStates states;
    tsl::robin_map<u64, const Files::ListedShader*> by_hash;
    SavedShaderHost host;
};

struct Options {
    u32 rounds = 20;
    bool reflection = true;
    bool coordinate_3d = false;
    /// Also try a 3D variant where coordinates look 3D, as the emulator's workers do.
    bool variants_3d = true;
    /// -2: the saved one; -1: each shader's lowest semantic; else this base.
    int semantic_base = -2;
    bool verbose = false;
};

/// What a translation came out as: its SPIR-V's hash and its specialisation, or nothing when
/// it failed a check.
struct Translated {
    u64 spv_hash;
    Shader::StageSpecialization spec;
    u64 flat_layout;
    bool saw_3d_coordinates;
};

std::optional<Translated> Translate(const Shader::Profile& profile, Shader::Pools& pools,
                                    Shader::Info& info, Shader::RuntimeInfo runtime_info,
                                    std::span<const u32> code, bool guessed) {
    try {
        const Common::SoftAsserts soft_asserts;
        Shader::Backend::Bindings binding{};
        const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
        if (guessed && !GuessTranslated(info, info.hw_stage)) {
            return std::nullopt;
        }
        const auto spv =
            Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
        return Translated{
            .spv_hash = XXH3_64bits(spv.data(), spv.size() * sizeof(u32)),
            .spec = Shader::StageSpecialization(info, runtime_info, profile, {}),
            .flat_layout = info.flat_layout,
            .saw_3d_coordinates = info.saw_3d_coordinates,
        };
    } catch (const Common::SoftAssertFailure&) {
        return std::nullopt;
    }
}

int SemanticBase(const Game& game, const Options& options) {
    return options.semantic_base == -2 ? game.list.semantic_base : options.semantic_base;
}

/// Translates every listed shader with a known stage and a struct, from its struct, `rounds`
/// times, timing translation alone. The emulator's precompile workers do the same work.
int Bench(const Game& game, const Options& options) {
    const auto& profile = game.host.profile;
    const ShaderHost host = game.host.Host();
    std::vector<const Files::ListedShader*> shaders;
    for (const auto& shader : game.list.shaders) {
        if (shader.stage != Files::NoStage && !shader.header.empty() &&
            PrecompilesStage(static_cast<HwStage>(shader.stage))) {
            shaders.push_back(&shader);
        }
    }
    std::ranges::sort(shaders, {}, &Files::ListedShader::hash);
    fmt::print("Translation bench: {} saved shaders with a known stage, {} rounds\n",
               shaders.size(), options.rounds);
    struct Timed {
        u64 ns;
        u64 hash;
        HwStage stage;
        size_t dwords;
    };
    std::vector<Timed> timed;
    tsl::robin_map<u64, size_t> timed_index;
    std::array<u64, 7> stage_ns{};
    std::array<u32, 7> stage_count{};
    Shader::Pools pools;
    Shader::TranslateTimes phases{};
    u64 emit_ns = 0;
    u64 total_ns = 0;
    u64 dwords = 0;
    u32 translated = 0;
    u32 unguessable = 0;
    for (u32 round = 0; round < options.rounds; ++round) {
        for (const auto* shader : shaders) {
            const auto hw_stage = static_cast<HwStage>(shader->stage);
            const auto sw_stage = SwStageOf(hw_stage);
            const auto runtime_info = GuessRuntimeInfo(host, hw_stage, sw_stage, shader->header,
                                                       SemanticBase(game, options));
            std::optional<Shader::Gcn::FetchShaderData> fetch;
            if (hw_stage == HwStage::Vertex) {
                fetch = GuessFetchShader(shader->header);
            }
            if (!runtime_info || (hw_stage == HwStage::Vertex && !fetch)) {
                unguessable += round == 0 ? 1 : 0;
                continue;
            }
            const Shader::ShaderParams params{
                .user_data = NoUserData,
                .code = shader->code,
                .hash = shader->hash,
            };
            Shader::Info info(hw_stage, sw_stage, params);
            info.guess_state = true;
            info.guess_coordinate_3d = options.coordinate_3d;
            info.guessed_fetch = fetch ? &*fetch : nullptr;
            auto runtime = *runtime_info;
            Shader::Backend::Bindings binding{};
            const auto before = Shader::ThreadTranslateTimes();
            const u64 start = NowNs();
            u64 emit_start = 0;
            u64 end = 0;
            try {
                const Common::SoftAsserts soft_asserts;
                const auto ir_program =
                    Shader::TranslateProgram(shader->code, pools, info, runtime, profile);
                if (!GuessTranslated(info, hw_stage)) {
                    throw Common::SoftAssertFailure{};
                }
                emit_start = NowNs();
                const auto spv =
                    Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, ir_program, binding);
                end = NowNs();
            } catch (const Common::SoftAssertFailure&) {
                unguessable += round == 0 ? 1 : 0;
                continue;
            }
            const auto& after = Shader::ThreadTranslateTimes();
            for (size_t i = 0; i < phases.ns.size(); ++i) {
                phases.ns[i] += after.ns[i] - before.ns[i];
            }
            emit_ns += end - emit_start;
            total_ns += end - start;
            dwords += shader->code.size();
            ++translated;
            const auto stage_index = static_cast<size_t>(hw_stage);
            stage_ns[stage_index] += end - start;
            ++stage_count[stage_index];
            if (const auto [it, added] = timed_index.try_emplace(shader->hash, timed.size());
                added) {
                timed.push_back({end - start, shader->hash, hw_stage, shader->code.size()});
            } else {
                timed[it->second].ns = std::min(timed[it->second].ns, end - start);
            }
        }
    }
    std::vector<std::pair<u64, std::string_view>> parts;
    for (size_t i = 0; i < phases.ns.size(); ++i) {
        parts.emplace_back(phases.ns[i],
                           Shader::TranslatePhaseName(static_cast<Shader::TranslatePhase>(i)));
    }
    parts.emplace_back(emit_ns, "SPIR-V emission");
    std::ranges::sort(parts, std::greater{});
    fmt::print("{} translations in {:.1f} ms ({:.1f} us per shader, {:.0f} us per 1000 code "
               "dwords), {} without a usable struct\n",
               translated, total_ns / 1e6, translated != 0 ? total_ns / 1e3 / translated : 0.0,
               dwords != 0 ? total_ns / 1e3 / (dwords / 1e3) : 0.0, unguessable);
    for (size_t i = 0; i < stage_ns.size(); ++i) {
        if (stage_count[i] != 0) {
            fmt::print("  {}: {} in {:.1f} ms\n", StageName(static_cast<HwStage>(i)),
                       stage_count[i], stage_ns[i] / 1e6);
        }
    }
    fmt::print("By phase:\n");
    for (const auto& [ns, name] : parts) {
        fmt::print("  {:<32} {:9.1f} ms {:5.1f}%\n", name, ns / 1e6,
                   100.0 * double(ns) / double(std::max<u64>(total_ns, 1)));
    }
    std::ranges::sort(timed, std::greater{}, &Timed::ns);
    fmt::print("Slowest:\n");
    for (size_t i = 0; i < std::min<size_t>(timed.size(), 10); ++i) {
        fmt::print("  {} {:#018x} {:6} dwords {:7.2f} ms\n", StageName(timed[i].stage),
                   timed[i].hash, timed[i].dwords, timed[i].ns / 1e6);
    }
    return 0;
}

/// Translates a recorded draw state the way the precompile workers replay it.
std::optional<Translated> TranslateRecorded(const Game& game, Shader::Pools& pools,
                                            const Files::ListedShader& shader,
                                            const DrawState& state) {
    const Shader::ShaderParams params{
        .user_data = state.user_data,
        .code = shader.code,
        .hash = shader.hash,
    };
    Shader::Info info{state.hw_stage, state.sw_stage, params};
    const Shader::Info::ReplayState replay{
        .flat_buf = state.flat_buf,
        .fetch = state.fetch ? &*state.fetch : nullptr,
        .vertex_buffers = state.vertex_buffers,
    };
    info.replay = &replay;
    return Translate(game.host.profile, pools, info, state.runtime_info, shader.code, false);
}

/// Counts "img3.type" as "img.type".
std::string KindOf(std::string_view part) {
    std::string kind;
    std::ranges::copy_if(part, std::back_inserter(kind),
                         [](char c) { return !std::isdigit(static_cast<unsigned char>(c)); });
    return kind;
}

/// For each recorded draw state of a shader with a struct, translates it as recorded (what the
/// draw really needed) and as the precompile workers guess it from the struct, and counts how
/// often the guess would have served the draw (same specialisation) and what it got wrong.
int Score(const Game& game, const Options& options) {
    const auto& profile = game.host.profile;
    const ShaderHost host = game.host.Host();
    Shader::Pools pools;
    struct StageScore {
        u32 states{};
        u32 no_struct{};
        u32 stale{};
        u32 truth_failed{};
        u32 unguessable{};
        u32 served{};
        u32 same_spirv{};
        /// With the draw's own registers, only resources guessed.
        u32 served_resources{};
        /// Misses that also got a 3D variant, and those it served (counted in served too).
        u32 tried_3d{};
        u32 served_3d{};
    };
    std::map<HwStage, StageScore> scores;
    std::map<std::string, u32> wrong;
    std::map<std::string, u32> wrong_resources;
    std::vector<u64> hashes;
    for (const auto& [hash, list] : game.states) {
        hashes.push_back(hash);
    }
    std::ranges::sort(hashes);
    for (const u64 hash : hashes) {
        const auto it = game.by_hash.find(hash);
        if (it == game.by_hash.end()) {
            continue;
        }
        const auto& shader = *it->second;
        for (const auto& blob : game.states.at(hash)) {
            const auto state = ParseDrawState(host, blob);
            if (!state) {
                continue;
            }
            auto& score = scores[state->hw_stage];
            ++score.states;
            const auto truth = TranslateRecorded(game, pools, shader, *state);
            if (!truth) {
                ++score.truth_failed;
                continue;
            }
            if (truth->flat_layout != state->flat_layout) {
                ++score.stale;
                continue;
            }
            if (shader.header.empty()) {
                ++score.no_struct;
                continue;
            }
            const auto guess_images =
                options.reflection
                    ? ReflectedImages(state->hw_stage, shader.header, shader.reflection)
                    : std::vector<Shader::Info::GuessedImage>{};
            const auto guess = [&](const Shader::RuntimeInfo& runtime,
                                   const Shader::Gcn::FetchShaderData* fetch, bool coordinate_3d) {
                const Shader::ShaderParams params{
                    .user_data = NoUserData,
                    .code = shader.code,
                    .hash = hash,
                };
                Shader::Info info{state->hw_stage, state->sw_stage, params};
                info.guess_state = true;
                info.guess_coordinate_3d = coordinate_3d;
                info.guessed_fetch = fetch;
                info.guessed_images = guess_images;
                return Translate(profile, pools, info, runtime, shader.code, true);
            };
            // Resources only: the draw's registers and fetch shader.
            if (const auto resources =
                    guess(state->runtime_info, state->fetch ? &*state->fetch : nullptr,
                          options.coordinate_3d)) {
                if (resources->spec == truth->spec) {
                    ++score.served_resources;
                } else {
                    for (const auto& part : SpecDifferences(truth->spec, resources->spec)) {
                        ++wrong_resources[KindOf(part)];
                    }
                }
            }
            // Everything from the struct, as a worker compiling ahead does.
            const auto runtime = GuessRuntimeInfo(host, state->hw_stage, state->sw_stage,
                                                  shader.header, SemanticBase(game, options));
            std::optional<Shader::Gcn::FetchShaderData> fetch;
            if (state->hw_stage == HwStage::Vertex) {
                fetch = GuessFetchShader(shader.header);
            }
            if (!runtime || (state->hw_stage == HwStage::Vertex && !fetch)) {
                ++score.unguessable;
                continue;
            }
            const auto full = guess(*runtime, fetch ? &*fetch : nullptr, options.coordinate_3d);
            if (!full) {
                ++score.unguessable;
                continue;
            }
            score.same_spirv += full->spv_hash == truth->spv_hash ? 1 : 0;
            if (full->spec == truth->spec) {
                ++score.served;
                continue;
            }
            if (options.variants_3d && !options.coordinate_3d && full->saw_3d_coordinates) {
                ++score.tried_3d;
                if (const auto as_3d = guess(*runtime, fetch ? &*fetch : nullptr, true);
                    as_3d && as_3d->spec == truth->spec) {
                    ++score.served;
                    ++score.served_3d;
                    continue;
                }
            }
            const auto parts = SpecDifferences(truth->spec, full->spec);
            for (const auto& part : parts) {
                ++wrong[KindOf(part)];
            }
            if (options.verbose) {
                fmt::print("  {} {:#018x}: {}\n", StageName(state->hw_stage), hash,
                           fmt::join(parts, ", "));
            }
        }
    }
    fmt::print("Guess score ({}{}, semantic base {}):\n",
               options.reflection ? "with reflection" : "reflection ignored",
               options.coordinate_3d ? ", 3D from coordinates" : "", SemanticBase(game, options));
    fmt::print("  stage  states  no struct  stale  truth fails  unguessable  served  same SPIR-V  "
               "served (draw registers)\n");
    StageScore total{};
    for (const auto& [stage, s] : scores) {
        fmt::print("  {:<5} {:7} {:10} {:6} {:12} {:12} {:7} {:12} {:10}\n", StageName(stage),
                   s.states, s.no_struct, s.stale, s.truth_failed, s.unguessable, s.served,
                   s.same_spirv, s.served_resources);
        total.states += s.states;
        total.no_struct += s.no_struct;
        total.stale += s.stale;
        total.truth_failed += s.truth_failed;
        total.unguessable += s.unguessable;
        total.served += s.served;
        total.same_spirv += s.same_spirv;
        total.served_resources += s.served_resources;
        total.tried_3d += s.tried_3d;
        total.served_3d += s.served_3d;
    }
    fmt::print("  {:<5} {:7} {:10} {:6} {:12} {:12} {:7} {:12} {:10}\n", "all", total.states,
               total.no_struct, total.stale, total.truth_failed, total.unguessable, total.served,
               total.same_spirv, total.served_resources);
    if (options.variants_3d && !options.coordinate_3d) {
        fmt::print("3D variants: {} misses also tried as 3D, {} of them served\n", total.tried_3d,
                   total.served_3d);
    }
    const auto print_kinds = [](const char* title, const std::map<std::string, u32>& kinds) {
        std::vector<std::pair<u32, std::string>> sorted;
        for (const auto& [kind, n] : kinds) {
            sorted.emplace_back(n, kind);
        }
        std::ranges::sort(sorted, std::greater{});
        fmt::print("{}:", title);
        for (const auto& [n, kind] : sorted) {
            fmt::print(" {} {},", kind, n);
        }
        fmt::print("{}\n", sorted.empty() ? " nothing" : "");
    };
    print_kinds("Wrong, guessing everything", wrong);
    print_kinds("Wrong, with the draw's registers", wrong_resources);
    return 0;
}

/// Translates every recorded draw state and prints a hash of each result, so two builds can be
/// checked to translate the same.
int States(const Game& game, const Options&) {
    Shader::Pools pools;
    std::vector<u64> hashes;
    for (const auto& [hash, list] : game.states) {
        hashes.push_back(hash);
    }
    std::ranges::sort(hashes);
    u64 all = 0;
    u32 count = 0;
    u32 failed = 0;
    for (const u64 hash : hashes) {
        const auto it = game.by_hash.find(hash);
        if (it == game.by_hash.end()) {
            continue;
        }
        u32 index = 0;
        for (const auto& blob : game.states.at(hash)) {
            const auto state = ParseDrawState(game.host.Host(), blob);
            if (!state) {
                continue;
            }
            const auto truth = TranslateRecorded(game, pools, *it->second, *state);
            fmt::print("{} {:#018x} {} {:#018x}\n", StageName(state->hw_stage), hash, index++,
                       truth ? truth->spv_hash : 0);
            if (truth) {
                all = HashCombine(all, truth->spv_hash);
                ++count;
            } else {
                ++failed;
            }
        }
    }
    fmt::print("{} states translated, {} failed, combined hash {:#018x}\n", count, failed, all);
    return 0;
}

int Usage() {
    std::fputs(
        "usage: shader_tool <cache folder> <serial> <command> [options]\n"
        "  bench [rounds]   time translating every listed shader from its struct (default 20)\n"
        "  score            score struct guesses against the recorded draw states\n"
        "  states           translate every recorded draw state, printing a hash of each\n"
        "options:\n"
        "  --no-reflection        ignore the PSSL reflection's image types\n"
        "  --coordinate-3d        guess untyped images as 3D from their coordinates\n"
        "  --no-3d-variants       don't also try 3D where coordinates look 3D\n"
        "  --semantic-base N      pixel shader semantic base (default: the saved one; -1 for\n"
        "                         each shader's lowest)\n"
        "  -v                     list each miss\n"
        "The cache folder holds <serial>_shaders.bin, <serial>_shader_states.bin and\n"
        "<serial>_shader_host.bin, which the emulator writes with the shader scan on.\n",
        stderr);
    return 2;
}

} // Anonymous namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        return Usage();
    }
    const std::filesystem::path folder = argv[1];
    const std::string serial = argv[2];
    const std::string_view command = argv[3];
    Options options;
    for (int i = 4; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-reflection") {
            options.reflection = false;
        } else if (arg == "--coordinate-3d") {
            options.coordinate_3d = true;
        } else if (arg == "--no-3d-variants") {
            options.variants_3d = false;
        } else if (arg == "--semantic-base" && i + 1 < argc) {
            options.semantic_base = std::atoi(argv[++i]);
        } else if (arg == "-v") {
            options.verbose = true;
        } else if (command == "bench" && std::isdigit(static_cast<unsigned char>(arg[0]))) {
            options.rounds = std::max(1, std::atoi(argv[i]));
        } else {
            return Usage();
        }
    }
    Game game;
    const auto host = ReadShaderHost(folder / (serial + "_shader_host.bin"));
    if (!host) {
        fmt::print(stderr,
                   "No usable {}_shader_host.bin in {}: run the game once with this "
                   "build and the shader scan on\n",
                   serial, folder.string());
        return 1;
    }
    game.host = *host;
    EmulatorSettings.SetDirectMemoryAccessEnabled(host->direct_memory_access);
    auto list = Files::ReadShaderList(folder / (serial + "_shaders.bin"));
    if (!list) {
        fmt::print(stderr, "No usable {}_shaders.bin in {}\n", serial, folder.string());
        return 1;
    }
    game.list = std::move(*list);
    if (auto states = Files::ReadShaderStates(folder / (serial + "_shader_states.bin"))) {
        game.states = std::move(*states);
    }
    for (const auto& shader : game.list.shaders) {
        game.by_hash.emplace(shader.hash, &shader);
    }
    u32 num_states = 0;
    for (const auto& [hash, list] : game.states) {
        num_states += static_cast<u32>(list.size());
    }
    fmt::print("{}: {} shaders, {} draw states of {} shaders, semantic base {}\n", serial,
               game.list.shaders.size(), num_states, game.states.size(), game.list.semantic_base);
    if (command == "bench") {
        return Bench(game, options);
    }
    if (command == "score") {
        return Score(game, options);
    }
    if (command == "states") {
        return States(game, options);
    }
    return Usage();
}
