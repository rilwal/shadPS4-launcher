// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <boost/container/small_vector.hpp>

#include "common/enum.h"
#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/texture_cache/image.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
struct StagingBufferRef;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

enum class ObtainBufferFlags {
    None = 0,
    IsWritten = 1 << 0,
    IsTexelBuffer = 1 << 1,
    IgnoreStreamBuffer = 1 << 2,
    InvalidateTextureCache = 1 << 3,
};
DECLARE_ENUM_FLAG_OPERATORS(ObtainBufferFlags)

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Marks memory the guest is about to write as modified from the CPU, ahead of its faults.
    void InvalidateMemoryAhead(VAddr device_addr, u64 size);

    /// Unmark a region as modified from the GPU.
    void UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool async = false,
                    bool assume_locks = false);

    /// Fenced readbacks: records copies of every GPU-modified buffer range into staging memory.
    /// They are written to guest memory on the scheduler's priority thread once the GPU reaches
    /// the current tick, so a priority operation queued after this runs after they've landed.
    /// The pages stay read-protected until then. Returns true when anything was recorded.
    bool LandGpuModifiedMemory();

    /// Flushes GPU modified ranges of the uncovered part of the edge pages of an image.
    void ReadEdgeImagePages(const Image& image);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(
        VAddr device_addr, u32 size, ObtainBufferFlags flags = ObtainBufferFlags::None);

    /// Attempts to obtain a buffer without modifying the cache contents.
    /// Returns a buffer holding the guest memory of an image. With device_local, memory that
    /// only the CPU wrote is copied into device memory first, so that a shader reading it in a
    /// scattered order (the detiler) doesn't read host memory across the bus.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size,
                                                                     bool device_local);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Mark region as modified from the GPU
    void MarkRegionAsGpuModified(VAddr addr, size_t size);

    /// Logs the readback stats gathered so far and starts a new window, e.g. at an A/B switch.
    void FlushReadbackStats();

    /// Readback profiling: records how a fence was signaled and time spent throttled.
    void RecordFence(bool deferred, u64 throttle_ns) {
        ++(deferred ? readback_stats.deferred_fences : readback_stats.immediate_fences);
        readback_stats.throttle_ns += throttle_ns;
    }

    /// Readback profiling: a fence that could have landed GPU-written memory but didn't,
    /// because the last landing was too recent.
    void RecordSkippedLanding() {
        ++readback_stats.skipped_landings;
    }

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    /// A set of GPU-modified ranges on their way back to guest memory (fenced readbacks).
    struct Landing {
        RangeSet ranges;
        /// Scheduler tick the copy was recorded in; it may not be submitted yet.
        u64 tick{};
        std::atomic<bool> done{false};
    };

    /// Unprotects the pages of finished landings that are still current, and drops them.
    void ProcessLandings();

    /// Waits until no landing in flight overlaps the region.
    void WaitForLandings(VAddr device_addr, u64 size);

    /// Logs readback_stats every few seconds, then resets them.
    void LogReadbackStats();

    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size, bool async);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    RangeSet gpu_modified_ranges;
    /// Pages the CPU has faulted on to read GPU data. Fenced readbacks land only these early.
    RangeSet cpu_read_ranges;
    /// Ranges taken by async downloads whose data isn't in guest memory yet (GPU thread only).
    RangeSet async_downloads;
    u32 num_async_downloads{};
    std::deque<std::shared_ptr<Landing>> landings;
    /// Host memory the landings are copied through: allocated at the head on the GPU thread,
    /// freed at the tail by the priority thread (both are running byte totals).
    static constexpr u64 LandingRingSize = 64_MB;
    std::unique_ptr<Buffer> landing_ring;
    u64 landing_ring_head{};
    std::atomic<u64> landing_ring_tail{};
    /// Why a synchronous readback was needed, for the readback stats.
    enum class ReadReason : u8 {
        /// The CPU read a page it hadn't read before.
        FirstRead,
        /// The CPU read a page it had read before, whose data hadn't landed.
        RepeatRead,
        /// The CPU wrote a page the GPU had written.
        Write,
        /// Anything else: unmaps, the texture cache, GC.
        Other,
    };
    ReadReason read_reason{ReadReason::Other};

    /// Precise or Fenced: GPU-written pages are protected until their data is in guest memory.
    bool tracks_gpu_pages{};

    /// Readback statistics, logged every few seconds.
    struct ReadbackStats {
        u64 sync_downloads{};
        u64 sync_download_bytes{};
        u64 landings{};
        u64 landing_bytes{};
        u64 stall_ns{};
        u64 throttle_ns{};
        u64 deferred_fences{};
        u64 immediate_fences{};
        u64 skipped_landings{};
        u64 full_ring_landings{};
        /// Stalling downloads and their stalls by why they were needed (ReadReason).
        std::array<u64, 4> sync_by_reason{};
        std::array<u64, 4> stall_ns_by_reason{};
        u64 last_log_ns{};
        u64 ab_switches{};
    } readback_stats;

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
