// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/fault_load.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

namespace {

/// Follows guest threads writing tracked memory in order (streaming data into a buffer), so
/// their write traps can be removed ahead of them: each fault in such a run unprotects twice as
/// many pages after it as the last, up to SHADPS4_FAULT_AHEAD_PAGES (0 turns it off). The pages
/// are only marked as written early, which costs an upload of data that didn't change at worst.
/// Runs that start where one started before (the same buffer refilled every frame) unprotect
/// as far as that one reached on their first fault (SHADPS4_FAULT_AHEAD_HISTORY=0 turns it off).
class FaultAhead {
public:
    /// The range to unprotect after a write fault at `addr`, empty if it isn't in a run.
    /// `predicted` says it came from an earlier run's length.
    std::pair<VAddr, u64> Next(VAddr addr, bool& predicted) {
        predicted = false;
        const u64 max_pages = MaxPages();
        if (max_pages == 0) {
            return {};
        }
        const VAddr page = PageManager::GetPageAddr(addr);
        const VAddr start = page + PageManager::GetPageSize();
        std::scoped_lock lock{mutex};
        ++tick;
        Run* run = std::ranges::find(runs, page, &Run::end);
        if (run == runs.end()) {
            // A new run, replacing the one used longest ago.
            run = std::ranges::min_element(runs, {}, &Run::last_use);
            *run = Run{.start = page, .end = start, .pages = 0, .last_use = tick};
            const History& seen = history[HistoryIndex(page)];
            if (!HistoryEnabled() || seen.start != page || seen.pages <= 1) {
                return {};
            }
            // Ahead as far as the last run from here went, after this page. Its next fault
            // carries on doubling from the most a fault takes.
            predicted = true;
            run->pages = max_pages;
            run->end = page + seen.pages * PageManager::GetPageSize();
            return {start, run->end - start};
        }
        run->pages = std::clamp<u64>(run->pages * 2, 1, max_pages);
        run->last_use = tick;
        run->end = start + run->pages * PageManager::GetPageSize();
        History& entry = history[HistoryIndex(run->start)];
        entry.start = run->start;
        entry.pages = static_cast<u32>(
            std::min<u64>((run->end - run->start) / PageManager::GetPageSize(), MaxHistoryPages));
        return {start, run->end - start};
    }

private:
    static u64 MaxPages() {
        static const u64 pages = [] {
            const char* env = std::getenv("SHADPS4_FAULT_AHEAD_PAGES");
            return env != nullptr ? std::strtoull(env, nullptr, 10) : u64{64};
        }();
        return pages;
    }

    static bool HistoryEnabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("SHADPS4_FAULT_AHEAD_HISTORY");
            return env == nullptr || env[0] != '0';
        }();
        return enabled;
    }

    static size_t HistoryIndex(VAddr page) {
        const u64 n = page / PageManager::GetPageSize();
        return (n ^ (n >> 12)) & (NumHistory - 1);
    }

    struct Run {
        /// Its first page, the one whose fault started it.
        VAddr start{};
        /// The first page the run hasn't unprotected yet, where its next fault lands.
        VAddr end{};
        u64 pages{};
        u64 last_use{};
    };
    /// How far the last run from a page reached, in pages from it (capped: unprotecting more
    /// than a run writes costs uploads of unchanged pages).
    struct History {
        VAddr start{};
        u32 pages{};
    };
    static constexpr u64 MaxHistoryPages = 256;
    static constexpr size_t NumHistory = 4096;
    std::mutex mutex;
    std::array<Run, 8> runs{};
    std::array<History, NumHistory> history{};
    u64 tick{};
};

FaultAhead fault_ahead;

/// Read faults whose load reads no GPU-written bytes are finished from guest memory
/// (Core::FaultLoad) instead of reading back the page. Read once: the handler runs often.
bool FaultLoadsEnabled() {
    static const bool enabled = EmulatorSettings.IsFaultLoadEmulation();
    return enabled;
}

std::atomic<u64> write_fault_count{};
std::atomic<u64> write_fault_ns{};
/// WriteFaultPhases summed over every handled write fault, and over the unprotecting ahead.
struct AtomicPhases {
    std::atomic<u64> region_lock_ns{};
    std::atomic<u64> page_lock_ns{};
    std::atomic<u64> protect_ns{};
    std::atomic<u64> texture_ns{};
    std::atomic<u64> protect_calls{};
    std::atomic<u64> protect_pages{};

    void Add(const WriteFaultPhases& p) {
        region_lock_ns.fetch_add(p.region_lock_ns, std::memory_order_relaxed);
        page_lock_ns.fetch_add(p.page_lock_ns, std::memory_order_relaxed);
        protect_ns.fetch_add(p.protect_ns, std::memory_order_relaxed);
        texture_ns.fetch_add(p.texture_ns, std::memory_order_relaxed);
        protect_calls.fetch_add(p.protect_calls, std::memory_order_relaxed);
        protect_pages.fetch_add(p.protect_pages, std::memory_order_relaxed);
    }

    WriteFaultPhases Take() {
        return {
            .region_lock_ns = region_lock_ns.exchange(0),
            .page_lock_ns = page_lock_ns.exchange(0),
            .protect_ns = protect_ns.exchange(0),
            .texture_ns = texture_ns.exchange(0),
            .protect_calls = protect_calls.exchange(0),
            .protect_pages = protect_pages.exchange(0),
        };
    }
};
AtomicPhases write_fault_phase_totals;
std::atomic<u64> ahead_count{};
std::atomic<u64> ahead_pages{};
std::atomic<u64> ahead_predicted{};
/// Handled read faults: all, and those on the GPU thread.
std::array<std::atomic<u64>, 2> read_fault_count{};
std::array<std::atomic<u64>, 2> read_fault_ns{};
std::mutex slow_write_mutex;
struct SlowWriteTotals {
    u32 count;
    u64 total_ns;
    WriteFaultPhases phases;
};
std::map<std::pair<u64, bool>, SlowWriteTotals> slow_write_sites;

/// Handles a write fault on tracked memory, unprotecting ahead of it too if the guest is
/// writing in order (not on the GPU thread, which may hold the trackers' locks).
bool HandleWriteFault(Vulkan::Rasterizer* rasterizer, VAddr addr, u64 size, bool is_gpu_thread,
                      void* context) {
    if (!rasterizer->IsMapped(addr, size)) {
        return false;
    }
    const u64 start = PerfStats::NowNs();
    bool predicted = false;
    auto [ahead, ahead_size] =
        is_gpu_thread ? std::pair<VAddr, u64>{} : fault_ahead.Next(addr, predicted);
    if (ahead_size != 0 && !rasterizer->IsMapped(ahead, ahead_size)) {
        ahead_size = 0;
    }
    write_fault_phases = WriteFaultPhases{.active = true};
    rasterizer->InvalidateFaultedMemory(addr, size, is_gpu_thread, ahead, ahead_size);
    const WriteFaultPhases phases = write_fault_phases;
    write_fault_phases.active = false;
    // A plain write fault takes a few microseconds; one that read the page back took far more.
    constexpr u64 SlowNs = 50'000;
    if (const u64 elapsed = PerfStats::NowNs() - start; elapsed > SlowNs) {
        std::scoped_lock lock{slow_write_mutex};
        auto& site = slow_write_sites[{Core::FaultInstructionAddress(context), is_gpu_thread}];
        ++site.count;
        site.total_ns += elapsed;
        site.phases.region_lock_ns += phases.region_lock_ns;
        site.phases.page_lock_ns += phases.page_lock_ns;
        site.phases.protect_ns += phases.protect_ns;
        site.phases.texture_ns += phases.texture_ns;
    }
    write_fault_phase_totals.Add(phases);
    if (ahead_size != 0) {
        const u64 pages = ahead_size / PageManager::GetPageSize();
        ahead_count.fetch_add(1, std::memory_order_relaxed);
        ahead_pages.fetch_add(pages, std::memory_order_relaxed);
        ahead_predicted.fetch_add(predicted ? 1 : 0, std::memory_order_relaxed);
        PerfStats::frames.AddCost(PerfStats::Cost::PagesAhead, pages);
    }
    const u64 elapsed = PerfStats::NowNs() - start;
    PerfStats::frames.AddFault(elapsed);
    write_fault_count.fetch_add(1, std::memory_order_relaxed);
    write_fault_ns.fetch_add(elapsed, std::memory_order_relaxed);
    return true;
}

} // namespace

bool RegionLockProfiling() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_LOCK_PROFILE");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

WriteFaultTotals TakeWriteFaultTotals() {
    return {
        .count = write_fault_count.exchange(0),
        .ns = write_fault_ns.exchange(0),
        .phases = write_fault_phase_totals.Take(),
        .ahead_count = ahead_count.exchange(0),
        .ahead_pages = ahead_pages.exchange(0),
        .ahead_predicted = ahead_predicted.exchange(0),
    };
}

ReadFaultTotals TakeReadFaultTotals() {
    return {read_fault_count[0].exchange(0), read_fault_ns[0].exchange(0),
            read_fault_count[1].exchange(0), read_fault_ns[1].exchange(0)};
}

std::vector<SlowWriteFaultSite> TakeSlowWriteFaultSites() {
    std::scoped_lock lock{slow_write_mutex};
    std::vector<SlowWriteFaultSite> sites;
    sites.reserve(slow_write_sites.size());
    for (const auto& [key, totals] : slow_write_sites) {
        sites.push_back({key.first, key.second, totals.count, totals.total_ns, totals.phases});
    }
    slow_write_sites.clear();
    return sites;
}

struct PageManager::Impl {
    struct PageState {
        /// Signed: region trackers change protection after releasing their locks, so a change
        /// can land before an earlier one it undoes, taking a count below zero for a moment.
        /// Adds and removes commute, and a count that isn't zero protects the page: below zero
        /// a track is still on its way, and until it lands writes fault and retry (the bits
        /// already say what they do), which is safe where unprotecting wouldn't be.
        s16 num_write_watchers;
        s16 num_read_watchers;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        s16 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            // Only a few changes can be in flight for a page at once.
            constexpr s16 Limit = 1024;
            if (update_read) {
                num_read_watchers += std::to_underlying(read_op);
                ASSERT_MSG(num_read_watchers > -Limit && num_read_watchers < Limit,
                           "Read watchers out of range: {}", num_read_watchers);
            }
            if (update_write) {
                num_write_watchers += std::to_underlying(write_op);
                ASSERT_MSG(num_write_watchers > -Limit && num_write_watchers < Limit,
                           "Write watchers out of range: {}", num_write_watchers);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            LockTimingWriteFault(locks[page], write_fault_phases.page_lock_ns);

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            LockTimingWriteFault(locks[base_page + page], write_fault_phases.page_lock_ns);

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            HandleWriteFault(rasterizer, addr, 1,
                             ptid == rasterizer->GetGpuCommandProcessorThreadId(), nullptr);

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        const bool timed = write_fault_phases.active || RegionLockProfiling();
        const u64 start = timed ? WriteFaultClockNs() : 0;
        impl.Protect(address, size, perms);
        if (timed) {
            const u64 elapsed = WriteFaultClockNs() - start;
            thread_protect_ns += elapsed;
            if (write_fault_phases.active) {
                write_fault_phases.protect_ns += elapsed;
                ++write_fault_phases.protect_calls;
                write_fault_phases.protect_pages += size / PageManager::GetPageSize();
            }
        }
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        const auto is_gpu_thread =
            std::this_thread::get_id() == rasterizer->GetGpuCommandProcessorThread();
        if (Common::IsWriteError(context)) {
            return HandleWriteFault(rasterizer, addr, size, is_gpu_thread, context);
        }
        const u64 start = VideoCore::PerfStats::NowNs();
        bool handled;
        bool served = false;
        if (FaultLoadsEnabled()) {
            // A load that reads no GPU-written bytes is finished here from guest memory, which
            // has the current data of everything else, instead of reading back the page. On the
            // GPU thread too: the emulator's own reads of guest memory fault as well.
            Core::FaultLoad load{};
            const bool decoded = Core::DecodeFaultLoad(context, load) && addr >= load.address &&
                                 addr < load.address + load.size;
            std::array<u8, 32> data;
            const bool overlapping = load.address < load.store_address + load.size &&
                                     load.store_address < load.address + load.size;
            if (decoded && load.op == Core::FaultLoad::Op::Copy && !overlapping) {
                handled = rasterizer->CopyMemoryForLoad(
                    addr, size, load.address, load.store_address, load.size, served, is_gpu_thread);
            } else {
                const bool supported = decoded && load.op != Core::FaultLoad::Op::Copy;
                handled = rasterizer->ReadMemoryForLoad(addr, size, supported ? load.address : 0,
                                                        supported ? load.size : 0, load.mnemonic,
                                                        data.data(), served, is_gpu_thread);
            }
            if (served && !Core::CompleteFaultLoad(context, load, data.data())) {
                // Can't happen after a successful decode; retrying faults again and reads back.
                served = false;
                handled = rasterizer->ReadMemory(addr, size, is_gpu_thread);
            }
        } else {
            handled = rasterizer->ReadMemory(addr, size, is_gpu_thread);
        }
        if (is_gpu_thread && handled) {
            rasterizer->NoteGpuThreadReadFault(Core::FaultInstructionAddress(context), served);
        }
        if (handled) {
            const u64 elapsed = VideoCore::PerfStats::NowNs() - start;
            VideoCore::PerfStats::frames.AddFault(elapsed);
            read_fault_count[0].fetch_add(1, std::memory_order_relaxed);
            read_fault_ns[0].fetch_add(elapsed, std::memory_order_relaxed);
            if (is_gpu_thread) {
                read_fault_count[1].fetch_add(1, std::memory_order_relaxed);
                read_fault_ns[1].fetch_add(elapsed, std::memory_order_relaxed);
            }
        }
        return handled;
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
