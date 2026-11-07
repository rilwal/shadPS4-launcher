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

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include "common/logging/log.h"
#include "common/types.h"

namespace VideoCore {

/// Cheap counters for profiling, gathered by one thread each and logged in windows.
namespace PerfStats {

[[nodiscard]] inline u64 SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define SHADPS4_PERF_TSC 1
namespace Detail {
/// The time stamp counter scaled to steady_clock nanoseconds. The cost scopes read the clock
/// twice per buffer and image lookup, thousands of times a frame, and clock_gettime cost 7% of
/// the GPU thread in Bloodborne; rdtsc is a few times cheaper.
struct TscClock {
    bool usable{};
    u64 tsc0{};
    u64 ns0{};
    double ns_per_tick{};

    static TscClock Calibrate() {
        TscClock clock;
        u32 eax{}, ebx{}, ecx{}, edx{};
        // Only an invariant TSC ticks at one rate on every core, in every power state.
        if (!__get_cpuid(0x80000007, &eax, &ebx, &ecx, &edx) || (edx & (1u << 8)) == 0) {
            return clock;
        }
        const u64 ns_start = SteadyNs();
        const u64 tsc_start = __rdtsc();
        u64 ns_end = ns_start;
        while (ns_end - ns_start < 20'000'000) {
            ns_end = SteadyNs();
        }
        const u64 tsc_end = __rdtsc();
        if (tsc_end <= tsc_start) {
            return clock;
        }
        clock.usable = true;
        clock.tsc0 = tsc_end;
        clock.ns0 = ns_end;
        clock.ns_per_tick = double(ns_end - ns_start) / double(tsc_end - tsc_start);
        return clock;
    }
};

inline const TscClock& Tsc() {
    static const TscClock clock = TscClock::Calibrate();
    return clock;
}
} // namespace Detail
#endif

[[nodiscard]] inline u64 NowNs() {
#ifdef SHADPS4_PERF_TSC
    const auto& tsc = Detail::Tsc();
    if (tsc.usable) {
        return tsc.ns0 + u64(double(__rdtsc() - tsc.tsc0) * tsc.ns_per_tick);
    }
#endif
    return SteadyNs();
}

/// Where the GPU thread's time goes, split so a stutter's log line can say what held it up.
/// Shaders is added by the pipeline cache; the rest by scopes (`ScopedCost`). Busy is the GPU
/// thread's time outside its idle waits, and what the named costs don't explain of it is command
/// translation itself.
enum class Cost : u32 {
    Shaders,      ///< Translating shaders, creating pipelines, waiting for compile workers.
    Textures,     ///< Finding and creating images, uploading and detiling them.
    Buffers,      ///< Obtaining buffers and syncing their memory from the guest.
    GpuWaits,     ///< Waiting for the host GPU to finish work (scheduler waits).
    Submits,      ///< Submitting command buffers to the host queue.
    Presents,     ///< Waiting for the presenter to flip (video out label waits).
    Flips,        ///< Preparing flipped frames for the presenter (copying, waiting for a frame).
    Dma,          ///< Buffer copies and fills the guest asked for, and DMA buffer syncs.
    Busy,         ///< GPU thread time not spent idle or spinning on guest waits.
    GuestWaits,   ///< Spinning on guest waits (wait reg mem, semaphores, CE/DE counters).
    Faults,       ///< Handling guest page faults on tracked memory, summed over every thread.
    NumFaults,    ///< How many of those faults (a count, not time).
    PagesAhead,   ///< Pages unprotected ahead of guests writing in order (a count).
    BufferBytes,  ///< Guest memory copied into GPU buffers by the GPU thread (bytes).
    ImageUploads, ///< Images uploaded from guest memory by the GPU thread (a count).
    ImageBytes,   ///< Their guest size (bytes).
    Draws,        ///< Draws recorded (a count).
    Dispatches,   ///< Dispatches recorded (a count).
    // Command translation for draws and dispatches, split only with SHADPS4_DRAW_COSTS=1; the
    // named costs above (shaders, textures, buffers) stay out of these.
    DrawChecks,  ///< Pending GPU operations, draw filters, render state set-up.
    Pipelines,   ///< Building pipeline keys and finding the pipeline and shader variants.
    Bindings,    ///< Reading the shaders' sharps and gathering their buffers and images.
    Targets,     ///< Finding render targets and starting rendering.
    Descriptors, ///< Writing the draw's descriptors.
    Recording,   ///< Vertex and index buffers, barriers, dynamic state and the Vulkan calls.
    Count,
};

[[nodiscard]] inline bool CostsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_FRAME_COSTS");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

/// Splits command translation for each draw into its steps. Off by default: the extra clock
/// reads cost about as much as the textures and buffers scopes already do.
[[nodiscard]] inline bool DrawCostsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_DRAW_COSTS");
        return env != nullptr && env[0] == '1';
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
    static constexpr size_t CostCount = static_cast<size_t>(Cost::Count);

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
    void AddShaderTime(u64 ns);

    void AddFault(u64 ns) {
        AddCost(Cost::Faults, ns);
        AddCost(Cost::NumFaults, 1);
    }

    /// A count or size from the GPU thread, like the timed costs.
    void AddGpuCount(Cost cost, u64 amount) {
        if (is_gpu_thread) {
            AddCost(cost, amount);
        }
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
            {
                std::scoped_lock lock{window_mutex};
                for (size_t i = 0; i < CostCount; ++i) {
                    window_costs[i] += previous_costs[i];
                }
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
        /// The GPU thread's costs summed over the window's frames.
        std::array<u64, CostCount> costs;
    };

    /// One line of costs, as the stutter log prints them.
    static std::string DescribeCosts(const std::array<u64, CostCount>& c) {
        const auto ms = [&](Cost cost) { return c[static_cast<size_t>(cost)] / 1e6; };
        u64 named = c[size_t(Cost::Shaders)] + c[size_t(Cost::Textures)] +
                    c[size_t(Cost::Buffers)] + c[size_t(Cost::GpuWaits)] +
                    c[size_t(Cost::Submits)] + c[size_t(Cost::Presents)] + c[size_t(Cost::Flips)] +
                    c[size_t(Cost::Dma)];
        u64 draw_steps = 0;
        for (size_t i = size_t(Cost::DrawChecks); i <= size_t(Cost::Recording); ++i) {
            draw_steps += c[i];
        }
        named += draw_steps;
        const u64 busy = c[size_t(Cost::Busy)];
        // Without the split, "other" is all of command translation.
        const std::string steps =
            draw_steps == 0
                ? std::string{}
                : fmt::format(", draw checks {:.1f}, pipelines {:.1f}, bindings {:.1f}, targets "
                              "{:.1f}, descriptors {:.1f}, recording {:.1f}",
                              ms(Cost::DrawChecks), ms(Cost::Pipelines), ms(Cost::Bindings),
                              ms(Cost::Targets), ms(Cost::Descriptors), ms(Cost::Recording));
        return fmt::format(
            "{} draws and {} dispatches; "
            "busy {:.1f} ms (shaders {:.1f}, textures {:.1f} for {} images of {:.1f} MB, "
            "buffers {:.1f} for {:.1f} MB, GPU "
            "waits {:.1f}, submits {:.1f}, present waits {:.1f}, flips {:.1f}, DMA {:.1f}{}, "
            "other {:.1f}), "
            "guest waits {:.1f} ms; page faults {} taking {:.1f} ms, {} pages "
            "unprotected ahead",
            c[size_t(Cost::Draws)], c[size_t(Cost::Dispatches)], busy / 1e6, ms(Cost::Shaders),
            ms(Cost::Textures), c[size_t(Cost::ImageUploads)],
            c[size_t(Cost::ImageBytes)] / 1048576.0, ms(Cost::Buffers),
            c[size_t(Cost::BufferBytes)] / 1048576.0, ms(Cost::GpuWaits), ms(Cost::Submits),
            ms(Cost::Presents), ms(Cost::Flips), ms(Cost::Dma), steps,
            busy > named ? (busy - named) / 1e6 : 0.0, ms(Cost::GuestWaits),
            c[size_t(Cost::NumFaults)], ms(Cost::Faults), c[size_t(Cost::PagesAhead)]);
    }

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
            window.costs = std::exchange(window_costs, {});
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
    std::array<u64, CostCount> window_costs{};
};

inline Frames frames;

/// Adds the time it's alive to a cost, on the GPU thread. Nested scopes pause the outer one, so
/// each nanosecond lands in one cost only (a buffer sync that waits on the GPU is a GPU wait).
class ScopedCost {
public:
    /// A detail scope only counts with SHADPS4_DRAW_COSTS=1.
    explicit ScopedCost(Cost cost_, bool detail = false) : cost{cost_} {
        if (!is_gpu_thread || !CostsEnabled() || (detail && !DrawCostsEnabled())) {
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

    /// Ends this cost and goes on counting the next one, for code that runs in steps. Only
    /// called while no scope nested in this one is alive.
    void Switch(Cost next) {
        if (!active) {
            return;
        }
        const u64 now = NowNs();
        frames.AddCost(cost, now - start);
        cost = next;
        start = now;
    }

    /// Takes time that was counted as another cost (shaders) out of the scope it happened in.
    static void Exclude(u64 ns) {
        if (current == nullptr) {
            return;
        }
        current->start = std::min(current->start + ns, NowNs());
    }

private:
    static inline thread_local ScopedCost* current = nullptr;
    Cost cost;
    bool active = false;
    ScopedCost* parent = nullptr;
    u64 start = 0;
};

inline void Frames::AddShaderTime(u64 ns) {
    if (is_gpu_thread) {
        AddCost(Cost::Shaders, ns);
        ScopedCost::Exclude(ns);
    }
}

} // namespace PerfStats

} // namespace VideoCore
