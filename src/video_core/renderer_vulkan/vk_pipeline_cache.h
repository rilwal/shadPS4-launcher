// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <variant>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_compiler.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_shader_objects.h"
#include "video_core/shader_scanner.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
        /// The program's hash combined with its SPIR-V's, so a variant has the same identity
        /// whatever order the variants were made in.
        u64 hash;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    /// Translated ahead of its first use with guessed draw state, as variant 0. Cleared at the
    /// first use, which is counted as served or not.
    bool prepared{};
    /// Variant 0's SPIR-V while prepared, registered with the cache if a draw uses it.
    std::vector<u32> prepared_spv;

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec, u64 hash) {
        modules.emplace_back(module, std::move(spec), hash);
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    /// Look-ahead: creates the pipeline for the draw or dispatch the registers describe now on
    /// the workers, ahead of the GPU thread reaching it. Returns true if it was new. Only
    /// does anything in the async modes that skip draws.
    bool PrefetchGraphicsPipeline();
    bool PrefetchComputePipeline();

    /// True once after a dispatch made a new compute pipeline on the GPU thread.
    bool TakeNewComputePipeline() {
        return std::exchange(new_compute_pipeline, false);
    }

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    const GraphicsPipeline* LookUpGraphicsPipeline(DrawIndirectParams params, bool prefetch);
    const ComputePipeline* LookUpComputePipeline(bool prefetch);

    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    /// Translates and creates a shader module; returns it with the variant's content hash.
    std::pair<vk::ShaderModule, u64> CompileModule(Shader::Info& info,
                                                   Shader::RuntimeInfo& runtime_info,
                                                   const std::span<const u32>& code,
                                                   size_t perm_idx,
                                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);
    /// A render target as the fragment shader is specialized on it.
    Shader::PsColorBuffer ShaderColorBuffer(const Shader::PsColorBuffer& target) const;

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    /// Logs where pipeline compiles spent their time, with game frame hitches.
    void FlushCompileStats(u64 now_ns);

    /// How new graphics pipelines get created (the async_pipelines setting).
    enum class AsyncMode : u32 {
        /// On the GPU thread, which waits.
        Off = 0,
        /// On the workers. Until the pipeline is ready, draws to targets the game drew in the
        /// last frames are skipped, and draws to new targets wait for it.
        Skip = 1,
        /// Unoptimized on the GPU thread, then optimized on the workers, replacing it.
        Tiered = 2,
        /// On the workers, skipping every draw until the pipeline is ready (for comparison:
        /// it can leave targets drawn once wrong for good).
        SkipAll = 3,
    };

    [[nodiscard]] static const char* AsyncModeName(AsyncMode mode);

    /// The mode for new graphics pipelines (setting, or the A/B timer).
    [[nodiscard]] AsyncMode GetAsyncMode() const;

    [[nodiscard]] bool UseAsyncPipelines() const {
        return GetAsyncMode() != AsyncMode::Off;
    }

public:
    /// Whether draws with a pending pipeline are skipped only for targets drawn every frame.
    [[nodiscard]] bool SkipsOnlyRedrawnTargets() const {
        return GetAsyncMode() == AsyncMode::Skip;
    }

    /// Whether every draw with a pending pipeline is skipped.
    [[nodiscard]] bool SkipsAnyPendingDraw() const {
        return GetAsyncMode() == AsyncMode::SkipAll;
    }

    /// Waits for a pending pipeline, for a draw that mustn't be skipped.
    void WaitUntilReady(const GraphicsPipeline& pipeline);

private:
    /// How the next new graphics pipeline gets created, with its stats callbacks.
    PipelineCompileOptions MakeCompileOptions(u64 pipeline_hash, u64 recompile_ns, AsyncMode mode);

    /// Waits a little for a background pipeline, within a budget per frame's worth of time, so
    /// quick ones are drawn at once.
    /// Returns the time waited.
    u64 WaitForPipeline(const GraphicsPipeline& pipeline);

    /// Time spent compiling, split into our recompiler and the driver. Logged every few
    /// seconds (GPU thread only).
    struct CompileStats {
        u64 graphics{};
        u64 compute{};
        u64 shaders{};
        /// Shaders compiled again as another variant of a program already seen.
        u64 permutations{};
        /// Of those, ones that only differ in where their bindings start.
        u64 binding_permutations{};
        u64 recompile_ns{};
        /// Driver time on the GPU thread, so not counting background creations.
        u64 driver_ns{};
        /// Longest the GPU thread spent on one pipeline.
        u64 worst_ns{};
        u64 background{};
        u64 skipped_draws{};
        u64 waited_ns{};
        /// Draws that waited for their pipeline, as they draw to a new target.
        u64 waited_draws{};
        /// Of those, ones whose creation the GPU thread took over from the queue.
        u64 ran_now{};
        /// Dispatches that waited for a compute pipeline made ahead on the workers.
        u64 waited_dispatches{};
        /// Pipelines made by the look-ahead, and how many of them a draw later found.
        u64 prefetched{};
        u64 prefetch_used{};
        u64 last_log_ns{};
    } compile_stats;
    /// Pipelines the look-ahead made that no draw or dispatch has found yet.
    tsl::robin_set<const Pipeline*> prefetched;
    bool new_compute_pipeline{};
    /// Recompile time of the shaders gathered for the pipeline being looked up.
    u64 pending_recompile_ns{};

    /// Totals from the workers, taken by the stats window.
    struct BackgroundStats {
        std::atomic<u64> created{};
        std::atomic<u64> driver_ns{};
        std::atomic<u64> worst_driver_ns{};
        /// From the draw that asked for a pipeline to the pipeline being ready.
        std::atomic<u64> worst_latency_ns{};
        std::atomic<u64> probes{};
        std::atomic<u64> probe_optimized_ns{};
        std::atomic<u64> probe_unoptimized_ns{};
    };
    BackgroundStats background_stats;

    u64 wait_credit_ns{};
    u64 wait_credit_ns_at{};
    bool ab_async{};
    u64 ab_switched_ns{};

public:
    /// Called at submits: logs compile_stats every few seconds.
    void TickCompileStats();

    /// Counts a draw skipped because its pipeline is still being created.
    void CountSkippedDraw() {
        ++compile_stats.skipped_draws;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    /// With shader objects: every guest shader compiled once, shared by the draws that use it.
    std::unique_ptr<ShaderObjects> shader_objects;
    std::unique_ptr<VideoCore::ShaderScanner> shader_scanner;
    /// Measurement for translating shaders ahead of their first draw: each program the scan
    /// found is translated once more with nothing known of the draw (no user data, so empty
    /// resources), and compared with what the draw needed. Only while SHADPS4_GUESS_CHECK=1.
    void CheckGuess(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                    const Shader::ShaderParams& params, Shader::RuntimeInfo& runtime_info,
                    const Shader::Backend::Bindings& start, const Shader::StageSpecialization& spec,
                    u64 perm_hash);
    /// The register state a shader's PSSL struct gives, for translating it before any draw
    /// uses it: the stage registers as written there, with the rest of the draw guessed.
    std::optional<Shader::RuntimeInfo> GuessRuntimeInfo(Shader::HwStage hw_stage,
                                                        Shader::SwStage sw_stage,
                                                        std::span<const u8> header) const;
    void LogGuessStats();
    /// The guess check for a vertex shader: its register state and fetch shader from its struct.
    void CheckVertexGuess(const Shader::ShaderParams& params, std::span<const u8> header,
                          const Shader::Backend::Bindings& start,
                          const Shader::StageSpecialization& spec, u64 perm_hash);
    static std::optional<Shader::Gcn::FetchShaderData> GuessFetchShader(std::span<const u8> header);
    /// Learns how pixel shader input semantics map to vertex outputs from a first use. GPU
    /// thread; the guess reads semantic_base from any thread.
    void LearnSemanticBase(u64 hash, const Shader::RuntimeInfo& runtime_info);
    std::atomic<int> semantic_base{-1};
    int semantic_base_candidate{-1};
    u32 semantic_base_votes{};
    u32 semantic_base_misses{};
    bool guess_check{};
    /// Each checked program's guessed SPIR-V hashes that missed the first variant, to see if a
    /// later variant matches them. Zero once matched.
    struct GuessedHashes {
        /// With the draw's registers.
        u64 resources{};
        /// With the registers from the shader struct.
        u64 full{};
    };
    tsl::robin_map<u64, GuessedHashes> guessed_hashes;
    struct GuessStats {
        u32 checked{};
        /// Same SPIR-V as the first variant.
        u32 matched{};
        /// Same specialisation as the first variant.
        u32 spec_matched{};
        /// Matched none at first, but a later variant.
        u32 matched_later{};
        /// The scan's copy of the code differed from the draw's.
        u32 code_differs{};
        /// Skipped: stages the guess can't translate yet (vertex shaders need the fetch
        /// shader, tessellation stages their constants).
        u32 skipped{};
        /// How many programs each part of the specialisation was guessed wrong in.
        std::map<std::string, u32> wrong;
        /// With the registers from the shader struct too.
        u32 full_matched{};
        u32 full_matched_later{};
        /// Found without a usable shader struct.
        u32 no_registers{};
        /// Parts the struct's registers got wrong, beyond the resources.
        std::map<std::string, u32> registers_wrong;
        /// Raw structs logged so far.
        u32 dumped{};
        u32 dumped_vertex{};
        u32 vertex_checked{};
        u32 vertex_matched{};
        u32 vertex_fetch_matched{};
        u32 vertex_unguessable{};
        std::map<std::string, u32> vertex_wrong;
        bool changed{};
    } guess_stats;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    /// Last, so its workers stop before the pipelines and the cache they use are destroyed.
    std::unique_ptr<PipelineCompiler> compiler;

    /// Translating the shaders the scan finds before their first use (shader_precompile_threads).
    /// Workers translate the newest found first, since a game tends to use what it just loaded;
    /// the GPU thread takes in the results before looking up a program.
    void StartPrecompile(u32 num_threads);
    void StopPrecompile();
    void RunPrecompile(std::stop_token stop);
    void PrecompileOne(u64 hash, Shader::HwStage hw_stage);
    /// Moves finished precompiles into the program and pipeline caches. GPU thread.
    void TakePrecompiled();
    /// Learns which binary type is which stage from programs the game uses. GPU thread.
    void NotePrecompileType(u64 hash, Shader::HwStage hw_stage);
    /// A prepared program's first use: whether its guess served the draw.
    void NotePreparedUse(Program& program, u64 hash, bool served,
                         const Shader::StageSpecialization& spec, u64 perm_hash);
    struct Precompiled {
        u64 hash;
        std::unique_ptr<Program> program;
        std::unique_ptr<ComputePipeline> compute;
        u64 perm_hash;
    };
    struct Precompile {
        std::mutex mutex;
        std::condition_variable_any cv;
        /// Waiting for a worker, newest last.
        std::vector<std::pair<u64, Shader::HwStage>> queue;
        /// Found shaders whose binary type isn't known as a stage yet.
        std::vector<std::pair<u64, u8>> untyped;
        /// Programs queued, translated or already used by the game.
        tsl::robin_set<u64> seen;
        /// Per binary type: the stage it was seen as and how often.
        std::array<Shader::HwStage, 16> type_stage;
        std::array<u32, 16> type_votes{};
        std::array<bool, 16> type_conflict{};
        size_t found_cursor{};
        std::vector<Precompiled> done;
        std::atomic<bool> has_done{};
        // For the compile summary, since the last one.
        u32 translated{};
        u32 shader_objects{};
        u32 compute_pipelines{};
        u64 worker_ns{};
        u32 served{};
        u32 missed{};
        u32 duplicate{};
        bool active{};
        std::vector<std::jthread> threads;
    } precompile;
};

} // namespace Vulkan
