// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/types.h"

namespace VideoCore {

/// Cheap counters for profiling, gathered by one thread each and logged in windows.
namespace PerfStats {

[[nodiscard]] inline u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Where the GPU thread's time goes, split so a stutter's log line can say what held it up.
/// Shaders is added by the pipeline cache; the rest by scopes (`ScopedCost`). Busy is the GPU
/// thread's time outside its idle waits, and what the named costs don't explain of it is command
/// translation itself.
enum class Cost : u32 {
    Shaders,    ///< Translating shaders, creating pipelines, waiting for compile workers.
    Textures,   ///< Finding and creating images, uploading and detiling them.
    Buffers,    ///< Obtaining buffers and syncing their memory from the guest.
    GpuWaits,   ///< Waiting for the host GPU to finish work (scheduler waits).
    Submits,    ///< Submitting command buffers to the host queue.
    Presents,   ///< Waiting for the presenter to flip (video out label waits).
    Busy,       ///< GPU thread time not spent idle or spinning on guest waits.
    GuestWaits, ///< Spinning on guest waits (wait reg mem, semaphores, CE/DE counters).
    Faults,     ///< Handling guest page faults on tracked memory, summed over every thread.
    NumFaults,  ///< How many of those faults (a count, not time).
    Count,
};

[[nodiscard]] inline bool CostsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_FRAME_COSTS");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

/// Set on the GPU thread, so scopes on other threads (the presenter flushes too) don't count.
inline thread_local bool is_gpu_thread = false;

/// Game frame pacing, recorded by the presenter at each new game frame.
///
/// Stutter is judged against the game's own recent pace: a frame is a stutter when it takes
/// more than twice the median of the frames before it, so a steady 30 FPS doesn't count and a
/// loading screen's pace doesn't hide a hitch in gameplay. Its stall is the time past that
/// median, which is how long the game was actually held up.
struct Frames {
    /// Frames longer than this count as hitches (two 60 Hz frames).
    static constexpr u64 HitchNs = 33'333'333;
    /// Frames the median looks back over, and how many it needs first.
    static constexpr size_t PaceFrames = 120;
    static constexpr size_t MinPaceFrames = 30;

    std::atomic<u64> count{};
    std::atomic<u64> hitches{};
    /// Game frames presented since boot; never reset.
    std::atomic<u64> total{};
    std::atomic<u64> worst_ns{};
    /// Since boot; never reset.
    std::atomic<u64> total_stutters{};
    std::atomic<u64> total_stall_ns{};

    /// GPU-thread time spent on shaders and pipelines (translating, creating, waiting for
    /// workers), added by the pipeline cache, so a stutter's log line says how much of it they
    /// explain.
    void AddShaderTime(u64 ns) {
        if (is_gpu_thread) {
            AddCost(Cost::Shaders, ns);
        }
    }

    void AddFault(u64 ns) {
        AddCost(Cost::Faults, ns);
        AddCost(Cost::NumFaults, 1);
    }

    void AddCost(Cost cost, u64 ns) {
        costs[static_cast<size_t>(cost)].fetch_add(ns, std::memory_order_relaxed);
    }

    void Record() {
        total.fetch_add(1, std::memory_order_relaxed);
        const u64 now = NowNs();
        if (last_ns != 0) {
            const u64 delta = now - last_ns;
            count.fetch_add(1, std::memory_order_relaxed);
            if (delta > HitchNs) {
                hitches.fetch_add(1, std::memory_order_relaxed);
            }
            u64 worst = worst_ns.load(std::memory_order_relaxed);
            while (delta > worst &&
                   !worst_ns.compare_exchange_weak(worst, delta, std::memory_order_relaxed)) {
            }
            u64 stall = 0;
            if (num_paced >= MinPaceFrames) {
                std::array<u64, PaceFrames> sorted;
                const size_t n = std::min(num_paced, PaceFrames);
                std::copy_n(pace.begin(), n, sorted.begin());
                std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.begin() + n);
                const u64 median = sorted[n / 2];
                if (delta > median * 2) {
                    stall = delta - median;
                    total_stutters.fetch_add(1, std::memory_order_relaxed);
                    total_stall_ns.fetch_add(stall, std::memory_order_relaxed);
                    // The GPU thread runs up to a frame behind, so its time over this frame
                    // and the one before.
                    std::array<u64, CostCount> current;
                    for (size_t i = 0; i < CostCount; ++i) {
                        current[i] = costs[i].load(std::memory_order_relaxed);
                    }
                    LOG_INFO(Render,
                             "Stutter at frame {}: {:.1f} ms against a median of {:.1f} ms; GPU "
                             "thread in it: {}; in the frame before: {}",
                             total.load(std::memory_order_relaxed), delta / 1e6, median / 1e6,
                             DescribeCosts(current), DescribeCosts(previous_costs));
                }
            }
            for (size_t i = 0; i < CostCount; ++i) {
                previous_costs[i] = costs[i].exchange(0, std::memory_order_relaxed);
            }
            pace[next_pace] = delta;
            next_pace = (next_pace + 1) % PaceFrames;
            ++num_paced;
            std::scoped_lock lock{window_mutex};
            window_us.push_back(static_cast<u32>(std::min<u64>(delta / 1000, ~0u)));
            if (stall != 0) {
                ++window_stutters;
                window_stall_ns += stall;
                window_over_50 += delta > 50'000'000 ? 1 : 0;
                window_over_100 += delta > 100'000'000 ? 1 : 0;
            }
        }
        last_ns = now;
    }

    struct Window {
        u64 count;
        u64 hitches;
        u64 worst_ns;
        /// Stutters as above, and how many of them were over 50 and 100 ms.
        u64 stutters;
        u64 stall_ns;
        u64 over_50;
        u64 over_100;
        /// Frame time percentiles.
        u64 p99_ns;
        u64 p999_ns;
        u64 total_stutters;
        u64 total_stall_ns;
    };

    /// Returns the counts since the last call and starts a new window.
    Window Take() {
        std::vector<u32> times;
        Window window{
            .count = count.exchange(0, std::memory_order_relaxed),
            .hitches = hitches.exchange(0, std::memory_order_relaxed),
            .worst_ns = worst_ns.exchange(0, std::memory_order_relaxed),
        };
        {
            std::scoped_lock lock{window_mutex};
            times.swap(window_us);
            window.stutters = std::exchange(window_stutters, 0);
            window.stall_ns = std::exchange(window_stall_ns, 0);
            window.over_50 = std::exchange(window_over_50, 0);
            window.over_100 = std::exchange(window_over_100, 0);
        }
        window.total_stutters = total_stutters.load(std::memory_order_relaxed);
        window.total_stall_ns = total_stall_ns.load(std::memory_order_relaxed);
        const auto percentile = [&](double p) -> u64 {
            if (times.empty()) {
                return 0;
            }
            const size_t i = std::min(times.size() - 1, size_t(p * double(times.size())));
            std::nth_element(times.begin(), times.begin() + i, times.end());
            return u64(times[i]) * 1000;
        };
        window.p99_ns = percentile(0.99);
        window.p999_ns = percentile(0.999);
        return window;
    }

private:
    static constexpr size_t CostCount = static_cast<size_t>(Cost::Count);

    static std::string DescribeCosts(const std::array<u64, CostCount>& c) {
        const auto ms = [&](Cost cost) { return c[static_cast<size_t>(cost)] / 1e6; };
        const u64 named = c[size_t(Cost::Shaders)] + c[size_t(Cost::Textures)] +
                          c[size_t(Cost::Buffers)] + c[size_t(Cost::GpuWaits)] +
                          c[size_t(Cost::Submits)] + c[size_t(Cost::Presents)];
        const u64 busy = c[size_t(Cost::Busy)];
        return fmt::format("busy {:.1f} ms (shaders {:.1f}, textures {:.1f}, buffers {:.1f}, GPU "
                           "waits {:.1f}, submits {:.1f}, present waits {:.1f}, other {:.1f}), "
                           "guest waits {:.1f} ms; page faults {} taking {:.1f} ms",
                           busy / 1e6, ms(Cost::Shaders), ms(Cost::Textures), ms(Cost::Buffers),
                           ms(Cost::GpuWaits), ms(Cost::Submits), ms(Cost::Presents),
                           busy > named ? (busy - named) / 1e6 : 0.0, ms(Cost::GuestWaits),
                           c[size_t(Cost::NumFaults)], ms(Cost::Faults));
    }

    std::array<std::atomic<u64>, CostCount> costs{};
    std::array<u64, CostCount> previous_costs{}; // presenter thread only
    u64 last_ns{};                               // presenter thread only
    std::array<u64, PaceFrames> pace{};          // presenter thread only
    size_t next_pace{};
    size_t num_paced{};
    std::mutex window_mutex;
    std::vector<u32> window_us;
    u64 window_stutters{};
    u64 window_stall_ns{};
    u64 window_over_50{};
    u64 window_over_100{};
};

inline Frames frames;

/// Adds the time it's alive to a cost, on the GPU thread. Nested scopes pause the outer one, so
/// each nanosecond lands in one cost only (a buffer sync that waits on the GPU is a GPU wait).
class ScopedCost {
public:
    explicit ScopedCost(Cost cost_) : cost{cost_} {
        if (!is_gpu_thread || !CostsEnabled()) {
            return;
        }
        active = true;
        const u64 now = NowNs();
        parent = current;
        if (parent != nullptr) {
            frames.AddCost(parent->cost, now - parent->start);
        }
        current = this;
        start = now;
    }
    ~ScopedCost() {
        if (!active) {
            return;
        }
        const u64 now = NowNs();
        frames.AddCost(cost, now - start);
        current = parent;
        if (parent != nullptr) {
            parent->start = now;
        }
    }
    ScopedCost(const ScopedCost&) = delete;
    ScopedCost& operator=(const ScopedCost&) = delete;

private:
    static inline thread_local ScopedCost* current = nullptr;
    Cost cost;
    bool active = false;
    ScopedCost* parent = nullptr;
    u64 start = 0;
};

} // namespace PerfStats

} // namespace VideoCore
