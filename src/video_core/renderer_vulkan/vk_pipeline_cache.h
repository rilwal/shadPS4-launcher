// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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
};

} // namespace Vulkan
