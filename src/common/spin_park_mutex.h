// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include "common/types.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Common {

/// A mutex for locks held a few microseconds at a time by many threads: a waiter spins for up
/// to SpinNs() before sleeping. Waking a sleeping waiter takes tens of microseconds on a desktop
/// whose idle cores are in deep sleep states, far longer than such a lock is held.
class SpinParkMutex {
public:
    SpinParkMutex() = default;
    SpinParkMutex(const SpinParkMutex&) = delete;
    SpinParkMutex& operator=(const SpinParkMutex&) = delete;

    void lock() {
        if (try_lock()) [[likely]] {
            return;
        }
        if (const u64 spin_ns = SpinNs(); spin_ns != 0) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::nanoseconds(spin_ns);
            u32 spins = 0;
            do {
                if (state.load(std::memory_order_relaxed) == Unlocked && try_lock()) {
                    return;
                }
                Pause();
            } while ((++spins & 63) != 0 || std::chrono::steady_clock::now() < deadline);
        }
        // Mark the lock contended, so unlock() wakes a sleeper.
        u32 previous = state.exchange(Contended, std::memory_order_acquire);
        while (previous != Unlocked) {
            state.wait(Contended, std::memory_order_relaxed);
            previous = state.exchange(Contended, std::memory_order_acquire);
        }
    }

    void unlock() {
        if (state.exchange(Unlocked, std::memory_order_release) == Contended) {
            state.notify_one();
        }
    }

    [[nodiscard]] bool try_lock() {
        u32 expected = Unlocked;
        return state.compare_exchange_strong(expected, Locked, std::memory_order_acquire,
                                             std::memory_order_relaxed);
    }

    /// How long a waiter spins: 50 us, or SHADPS4_LOCK_SPIN_US (0 sleeps at once).
    static u64 SpinNs() {
        static const u64 ns = [] {
            const char* env = std::getenv("SHADPS4_LOCK_SPIN_US");
            return (env != nullptr ? std::strtoull(env, nullptr, 10) : 50) * 1000;
        }();
        return ns;
    }

private:
    static void Pause() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    }

    static constexpr u32 Unlocked = 0;
    static constexpr u32 Locked = 1;
    static constexpr u32 Contended = 2;
    std::atomic<u32> state{Unlocked};
};

} // namespace Common
