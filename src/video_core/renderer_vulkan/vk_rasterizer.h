// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>

#include "common/ab_switch.h"
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "shader_recompiler/runtime_buffer_format.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_resource_heaps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    /// Look-ahead: makes the pipeline the next draw or dispatch in the registers will use, on
    /// the workers. Returns true if it was new.
    bool PrefetchDraw();
    bool PrefetchDispatch();

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Removes buffer write traps ahead of a guest that's writing memory in order.
    void InvalidateMemoryAhead(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// ReadMemory for a read fault whose load the handler decoded (BufferCache::ReadMemoryForLoad).
    /// `served` says the load's bytes were copied to `data` instead of read back.
    bool ReadMemoryForLoad(VAddr addr, u64 size, VAddr load_addr, u32 load_size, u16 mnemonic,
                           u8* data, bool& served, bool assume_locks);
    /// The same for a movs copying `bytes` from `src` to `dst` (BufferCache::CopyMemoryForLoad).
    bool CopyMemoryForLoad(VAddr addr, u64 size, VAddr src, VAddr dst, u64 bytes, bool& served,
                           bool assume_locks);
    /// BufferCache::NoteGpuThreadReadFault.
    void NoteGpuThreadReadFault(u64 instruction, bool served) {
        buffer_cache.NoteGpuThreadReadFault(instruction, served);
    }
    bool IsMapped(VAddr addr, u64 size);
    /// Whether any byte of the range is mapped.
    bool IsAnyMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    /// Fenced readbacks: copies what the GPU wrote since the last fence back to guest memory
    /// and queues `signal` (the fence's label write and interrupt) to run once it has landed,
    /// so the guest only sees the fence once the data is there, as on hardware. With
    /// `throttle`, also keeps the command processor at most a couple of such fences ahead.
    /// When nothing was written since the last fence and no signal is pending, it signals
    /// right away instead. Returns false (without signaling) when readbacks aren't Fenced.
    bool DeferFenceSignal(Common::UniqueFunction<void>&& signal, bool throttle);

    /// Submits pending work when a deferred fence signal is waiting on it. The command
    /// processor calls this before it waits, as what it waits for may be that signal.
    void FlushFenceSignals();

    /// Whether fences currently use Fenced readbacks (see the A/B switch).
    [[nodiscard]] bool IsFencedReadbacks() const {
        return readbacks_ab ? Common::AbSwitch::IsB() : fenced_readbacks;
    }

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    /// Binds the pipeline, or its shader objects and their baked state.
    void BindGraphics(const GraphicsPipeline* pipeline);
    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    /// What FilterDraw would return, without doing the work it does for the draws it takes.
    bool WouldDraw() const;

    /// Decides what to do with a draw whose pipeline is still being created: true to skip it.
    /// Skips only when every target it draws to was also drawn in one of the last frames, so
    /// the game redraws it soon; otherwise waits, as a target drawn once (a baked texture)
    /// would stay wrong.
    bool SkipPendingDraw(const GraphicsPipeline* pipeline);

    /// Records the frame in which each of the draw's targets was drawn.
    void NoteDrawTargets(const GraphicsPipeline* pipeline);

    /// Color and depth target addresses of the current draw.
    boost::container::static_vector<VAddr, AmdGpu::NUM_COLOR_BUFFERS + 2> DrawTargets(
        const GraphicsPipeline* pipeline) const;

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    /// Binds what BindResources gathered, through the descriptor heap or the pipeline's layout.
    void CommitResources(const Pipeline* pipeline);

    /// A buffer descriptor's range; a null buffer makes a null descriptor.
    void AddBufferInfo(const VideoCore::Buffer* buffer, u64 offset, u64 size);
    /// An image descriptor's view; a null view makes a null descriptor.
    void AddImageInfo(const VideoCore::ImageView* view, vk::ImageLayout layout);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    /// The last two game frames in which something was drawn to a target address.
    struct TargetFrames {
        u64 last{};
        u64 previous{};
    };
    tsl::robin_map<VAddr, TargetFrames> target_frames;
    /// Targets already noted in target_frames this frame, so repeats skip the map.
    boost::container::static_vector<VAddr, 16> noted_targets;

    /// Shader::DescribeRuntimeBufferFormat, cached by the bits of the sharp it reads.
    std::optional<Shader::RuntimeBufferFormat> DescribeBufferFormat(const AmdGpu::Buffer& vsharp);
    /// dst_sel_x..w, num_format and data_format: the low 19 bits of the sharp's last dword.
    static constexpr u32 FormatKeyMask = (1u << 19) - 1;
    static constexpr u32 FormatKeyValid = 1u << 31;
    struct BufferFormatEntry {
        u32 key{};
        std::optional<Shader::RuntimeBufferFormat> format;
    };
    std::array<BufferFormatEntry, 64> buffer_formats{};
    u64 noted_frame{};

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    /// Set when shader resources go through VK_EXT_descriptor_heap. Declared before the caches
    /// so that it outlives the samplers that give their heap slots back to it.
    std::unique_ptr<ResourceHeaps> heaps;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    /// Readbacks Precise or Fenced: the two can be A/B switched live.
    bool readbacks_ab{};
    bool fenced_readbacks{};
    u64 last_ab_switch_ns{};
    std::atomic<u32> deferred_signals_in_flight{0};
    std::chrono::steady_clock::time_point last_landing{};
    u64 unsubmitted_signal_tick{};
    std::chrono::steady_clock::time_point unsubmitted_signal_since{};
    std::atomic<u32> throttled_fences_in_flight{0};

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    // Kept next to buffer_infos and image_infos for the heap.
    boost::container::static_vector<vk::DeviceAddress, Shader::NUM_BUFFERS> buffer_addresses;
    boost::container::static_vector<const VideoCore::ImageView*, Shader::NUM_IMAGES> image_views;
    boost::container::static_vector<u32, Shader::NUM_IMAGES> sampler_slots;
    boost::container::static_vector<HeapStageWrites, Shader::HeapPushData::NumStages> heap_stages;

    bool attachment_feedback_loop{};
    /// The pipeline whose shader objects are bound, while nothing has undone them.
    const GraphicsPipeline* bound_shaders{};
    bool needs_barrier{};
};

} // namespace Vulkan
