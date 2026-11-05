// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "common/types.h"

namespace VideoCore {

/// Cheap counters for profiling, gathered by one thread each and logged in windows.
namespace PerfStats {

[[nodiscard]] inline u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Game frame pacing, recorded by the presenter at each new game frame.
struct Frames {
    /// Frames longer than this count as hitches (two 60 Hz frames).
    static constexpr u64 HitchNs = 33'333'333;

    std::atomic<u64> count{};
    std::atomic<u64> hitches{};
    /// Game frames presented since boot; never reset.
    std::atomic<u64> total{};
    std::atomic<u64> worst_ns{};
    u64 last_ns{}; // presenter thread only

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
        }
        last_ns = now;
    }

    struct Window {
        u64 count;
        u64 hitches;
        u64 worst_ns;
    };

    /// Returns the counts since the last call and starts a new window.
    Window Take() {
        return Window{
            .count = count.exchange(0, std::memory_order_relaxed),
            .hitches = hitches.exchange(0, std::memory_order_relaxed),
            .worst_ns = worst_ns.exchange(0, std::memory_order_relaxed),
        };
    }
};

inline Frames frames;

} // namespace PerfStats

} // namespace VideoCore
