// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <functional>
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
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/shader_guess.h"
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
    /// Translated ahead of its first use, with guessed or recorded draw state. Cleared at the
    /// first use, which is counted as served or not.
    bool prepared{};
    /// The prepared variants' SPIR-V by variant hash, registered with the cache if a draw uses
    /// one.
    std::vector<std::pair<u64, std::vector<u32>>> prepared_spv;

    /// The inputs of the last draw's variant lookup, so a draw whose inputs match skips building
    /// and comparing the specialisation (see SpecKey).
    struct LastLookup {
        u32 module = UINT32_MAX;
        Shader::Backend::Bindings start{};
        Shader::RuntimeInfo runtime_info{};
        /// The fetch shader and the masked sharps (see SpecKey).
        boost::container::small_vector<u64, 48> key;
    };
    LastLookup last_lookup;

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec, u64 hash) {
        modules.emplace_back(module, std::move(spec), hash);
    }
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

    /// Whether found shaders are compiled early; tiling pipelines are then prepared ahead too.
    bool PreparesAhead() const {
        return precompile.active;
    }
    /// Tiling pipelines earlier runs used (packed TileManager keys), once they're loaded.
    std::vector<u64> SavedTilingKeys();
    void NoteTilingKey(u64 key);
    auto& GetProfile() const {
        return profile;
    }

private:
    const GraphicsPipeline* LookUpGraphicsPipeline(DrawIndirectParams params, bool prefetch);
    const ComputePipeline* LookUpComputePipeline(bool prefetch);

    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    /// A draw's shader parameters from its program registers (AmdGpu::GetParams), reading the
    /// shader's header through guest memory's backing where no GPU bytes are pending there.
    template <typename Program>
    Shader::ShaderParams ProgramParams(const Program& pgm) const;
    /// The same, empty where the program has no binary info.
    template <typename Program>
    std::optional<Shader::ShaderParams> TryProgramParams(const Program& pgm) const;
    /// Whether every program the graphics registers enable has its binary info. Look-ahead
    /// reads registers set past waits, whose shaders the game may not have loaded yet.
    bool GraphicsProgramsLoaded() const;

public:
    /// Whether any byte of a range has GPU data not yet in guest memory (the buffer cache's).
    void SetGpuBytesPending(std::function<bool(VAddr, u64)> func) {
        gpu_bytes_pending = std::move(func);
    }

private:
    std::function<bool(VAddr, u64)> gpu_bytes_pending;

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    /// Translates and creates a shader module; returns it with the variant's content hash.
    std::pair<vk::ShaderModule, u64> CompileModule(Shader::Info& info,
                                                   Shader::RuntimeInfo& runtime_info,
                                                   std::span<const u32> guest_code, size_t perm_idx,
                                                   Shader::Backend::Bindings& binding);
    /// Shader code to translate: a copy read through the backing where that's safe on the GPU
    /// thread, else the guest code itself.
    std::span<const u32> BackingCode(std::span<const u32> code);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

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
        /// Where the GPU thread's recompile time went: translation phase by phase, then
        /// SPIR-V emission, shader module creation and the rest (hashing, registering).
        Shader::TranslateTimes translate{};
        u64 emit_ns{};
        u64 module_ns{};
        u64 code_dwords{};
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
    /// Learns how pixel shader input semantics map to vertex outputs from a first use. GPU
    /// thread; the guess reads semantic_base from any thread.
    void LearnSemanticBase(u64 hash, const Shader::RuntimeInfo& runtime_info);
    std::atomic<int> semantic_base{-1};
    int semantic_base_candidate{-1};
    u32 semantic_base_votes{};
    u32 semantic_base_misses{};
    /// Draws whose pixel inputs the fallback guess (each shader's lowest semantic as the base)
    /// would have got right, and wrong. For the log.
    u32 semantic_min_right{};
    u32 semantic_min_wrong{};
    bool guess_check{};
    /// Draws whose variant lookup inputs match the program's last draw reuse its variant.
    /// SHADPS4_SPEC_MEMO=0 turns that off; SHADPS4_SPEC_MEMO_CHECK=1 also looks the variant up
    /// the full way on each reuse and logs any difference.
    bool spec_memo = true;
    bool spec_memo_check{};
    struct {
        u64 hits{};
        u64 misses{};
        u64 mismatches{};
    } spec_memo_stats;
    /// The draw's key and fetch shader, built for the comparison.
    boost::container::small_vector<u64, 48> spec_key;
    Shader::Gcn::FetchShaderData spec_fetch;
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
    ShaderHost host{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    /// Each stage's specialisation for the draw being looked up, refilled per draw.
    std::array<Shader::StageSpecialization, MaxShaderStages> draw_specs{};
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
    /// Moves finished precompiles into the program and pipeline caches. GPU thread.
    void TakePrecompiled();
    /// Waits a little for a worker compiling the program, so its first use can take that.
    void JoinPrecompile(u64 hash);
    /// Queues the pixel shaders held until their inputs could be guessed.
    void ReleaseHeldPrecompiles();
    /// The same, under the precompile lock.
    void RequeueHeldLocked();
    /// Learns which binary type is which stage from programs the game uses. GPU thread.
    void NotePrecompileType(u64 hash, Shader::HwStage hw_stage);
    /// A prepared program's first use: whether its guess served the draw.
    void NotePreparedUse(Program& program, u64 hash, bool served,
                         const Shader::StageSpecialization& spec, u64 perm_hash);
    struct Precompiled {
        u64 hash;
        std::unique_ptr<Program> program;
        /// Compute pipelines by variant hash.
        std::vector<std::pair<u64, std::unique_ptr<ComputePipeline>>> computes;
    };
    struct Precompile {
        std::mutex mutex;
        std::condition_variable_any cv;
        struct Job {
            u64 hash;
            Shader::HwStage stage;
            /// Where the scan found it, 0 if not known.
            u64 addr;
            /// When an earlier run first used it (ShaderScanner::Shader::use_order).
            u32 use_order;
        };
        /// Waiting for a worker, newest last.
        std::vector<Job> queue;
        /// Programs queued, translated or already used by the game.
        tsl::robin_set<u64> seen;
        /// Programs the game has used.
        tsl::robin_set<u64> used;
        /// Shaders earlier runs used go first, in the order they were first used. Then whether
        /// workers take the queued shader found nearest the latest first uses, or else the
        /// newest found.
        bool order_near{true};
        /// Per binary type: the stage it's compiled early as, and whether a first use showed
        /// otherwise.
        std::array<Shader::HwStage, 16> type_stage;
        std::array<bool, 16> type_conflict{};
        size_t found_cursor{};
        std::vector<Precompiled> done;
        std::atomic<bool> has_done{};
        /// Pixel shaders with inputs found before the semantic base was learned.
        std::vector<u64> held;
        /// When the first was held, and whether holding has stopped because learning took
        /// too long (they're then compiled with the fallback guess).
        u64 first_held_ns{};
        u64 hold_ns{};
        std::atomic<bool> hold_expired{};
        /// Programs a worker is compiling now, and a signal for each one it finishes.
        tsl::robin_set<u64> in_flight;
        std::condition_variable finished;
        // For the compile summary, since the last one.
        u32 translated{};
        u32 shader_objects{};
        u32 compute_pipelines{};
        u64 worker_ns{};
        u32 served{};
        u32 missed{};
        u32 duplicate{};
        /// Queued shaders the game used before a worker got to them.
        u32 skipped_used{};
        /// Shaders whose guessed translation failed, so they wait for their draw.
        u32 rejected{};
        /// Shaders compiled early from recorded draw states, and the variants made.
        u32 replayed{};
        /// Guessed shaders whose image types came from their reflection.
        u32 reflected{};
        bool use_reflection{true};
        /// Info::guess_coordinate_3d, from SHADPS4_GUESS_COORDINATE_3D=1.
        bool coordinate_3d{};
        /// Whether shaders whose coordinates look 3D also get a 3D variant (off with
        /// SHADPS4_GUESS_COORDINATE_3D=0), and how many did.
        bool coordinate_3d_variants{true};
        u32 guessed_3d{};
        u32 replayed_variants{};
        /// First uses that waited for the worker compiling them, and how long.
        u32 joined{};
        u64 joined_ns{};
        bool active{};
        std::vector<std::jthread> threads;
    } precompile;
    void PrecompileOne(const Precompile::Job& job);
    /// Compiles one early variant's SPIR-V and builds its driver part, adding it to `program`
    /// and `done`. Returns whether it made a shader object.
    bool PrepareVariant(Program& program, Precompiled& done, const Shader::Info& info,
                        Shader::StageSpecialization&& spec, std::vector<u32>&& spv);
    /// Compiles a shader early for each draw state recorded at its first uses in earlier runs.
    /// Returns false when none of them could be used.
    bool PrecompileRecorded(const Precompile::Job& job,
                            const VideoCore::ShaderScanner::Shader& found,
                            const std::vector<std::vector<u8>>& states);
    /// A recorded draw state, built again for this build.
    /// Records the draw state a first use was translated for, for PrecompileRecorded next time.
    void RecordDrawState(const Shader::Info& info, const Shader::RuntimeInfo& runtime_info,
                         const Shader::Gcn::FetchShaderData& fetch);
    /// A queue entry for a found shader, with where it was found and when it was first used.
    Precompile::Job MakePrecompileJob(u64 hash, Shader::HwStage stage);
    /// Index of the queued shader a worker should compile next. Under the precompile lock.
    size_t PickPrecompileJob();
};

} // namespace Vulkan
