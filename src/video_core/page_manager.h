// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

struct UffdImpl;
struct SignalImpl;

enum class PageOp : s8 {
    None = 0,
    Track = 1,
    Untrack = -1,
};

/// Where a write fault's handling spent its time, on the faulting thread while it's handled.
/// For the slow write fault log.
struct WriteFaultPhases {
    bool active{};
    u64 region_lock_ns{};
    u64 page_lock_ns{};
    u64 protect_ns{};
    u64 texture_ns{};
};
inline thread_local WriteFaultPhases write_fault_phases;

inline u64 WriteFaultClockNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Locks the mutex, adding any wait to `wait_ns` while a write fault is handled on this thread.
template <typename Mutex>
void LockTimingWriteFault(Mutex& mutex, u64& wait_ns) {
    if (!write_fault_phases.active || mutex.try_lock()) [[likely]] {
        if (!write_fault_phases.active) {
            mutex.lock();
        }
        return;
    }
    const u64 start = WriteFaultClockNs();
    mutex.lock();
    wait_ns += WriteFaultClockNs() - start;
}

/// Region tracker lock holds over 20 us, by what held the lock (SHADPS4_LOCK_PROFILE=1), for
/// the readback stats log: write faults waiting on that lock are slow.
enum class RegionLockOp : u8 {
    CpuSet,
    CpuClear,
    GpuSet,
    GpuClear,
    CpuSetGpuClear,
    CpuClearGpuSet,
    UploadRanges,
    UploadRangesWritten,
    DownloadRanges,
    Explicit,
    Other,
    Count,
};
struct RegionLockHolds {
    std::atomic<u64> count;
    std::atomic<u64> hold_ns;
    std::atomic<u64> protect_ns;
    std::atomic<u64> max_ns;
};
inline std::array<RegionLockHolds, static_cast<size_t>(RegionLockOp::Count)> region_lock_holds{};
/// Time this thread has spent in mprotect, always counted while lock profiling is on.
inline thread_local u64 thread_protect_ns{};
bool RegionLockProfiling();

/// Times one hold of a region tracker lock, from construction (after locking) to Finish().
class RegionLockHoldTimer {
public:
    RegionLockHoldTimer() {
        if (RegionLockProfiling()) [[unlikely]] {
            start = WriteFaultClockNs();
            protect_start = thread_protect_ns;
        }
    }
    void Finish(RegionLockOp op) {
        if (start == 0) [[likely]] {
            return;
        }
        constexpr u64 LongNs = 20'000;
        const u64 held = WriteFaultClockNs() - start;
        if (held > LongNs) {
            auto& holds = region_lock_holds[static_cast<size_t>(op)];
            holds.count.fetch_add(1, std::memory_order_relaxed);
            holds.hold_ns.fetch_add(held, std::memory_order_relaxed);
            holds.protect_ns.fetch_add(thread_protect_ns - protect_start,
                                       std::memory_order_relaxed);
            u64 max = holds.max_ns.load(std::memory_order_relaxed);
            while (held > max && !holds.max_ns.compare_exchange_weak(max, held)) {
            }
        }
    }

private:
    u64 start{};
    u64 protect_start{};
};

/// A write fault that took long enough to have read memory back first, by faulting instruction:
/// how many, and their total time and its parts.
struct SlowWriteFaultSite {
    u64 instruction;
    bool gpu_thread;
    u32 count;
    u64 total_ns;
    WriteFaultPhases phases;
};

/// All handled write faults since the last call and their total time, cleared.
std::pair<u64, u64> TakeWriteFaultTotals();

/// The slow write fault sites since the last call, cleared. For the readback stats log.
std::vector<SlowWriteFaultSite> TakeSlowWriteFaultSites();

class PageManager {
    static constexpr size_t PM_PAGE_BITS = 12;
    static constexpr size_t PM_PAGE_SIZE = 1ULL << PM_PAGE_BITS;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// Updates watches in the pages touching the specified region.
    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const;

    /// Updates watches in the pages touching the inclusive bounds using a mask.
    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

    static constexpr size_t GetPageSize() {
        return PM_PAGE_SIZE;
    }

private:
    friend struct UffdImpl;
    friend struct SignalImpl;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
