// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
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

/// A write fault that took long enough to have read memory back first, by faulting instruction:
/// how many, and their total time and its parts.
struct SlowWriteFaultSite {
    u64 instruction;
    bool gpu_thread;
    u32 count;
    u64 total_ns;
    WriteFaultPhases phases;
};

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
