// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/logging/log.h"
#include "common/types.h"

namespace Common {

/// In-process A/B testing. One experiment registers two named variants, and a hotkey (or a
/// timer) flips between them while the game runs, so both are measured in the same process,
/// scene and warm caches. That has far less variance than comparing separate runs.
class AbSwitch {
public:
    /// Registers the experiment. Only the first registration is kept; returns whether this one
    /// was. `start_b` selects the variant that is active first.
    static bool Register(const char* label_a, const char* label_b, bool start_b) {
        const char* expected = nullptr;
        if (!a.compare_exchange_strong(expected, label_a)) {
            return false;
        }
        b.store(label_b);
        is_b.store(start_b);
        LOG_INFO(Common, "A/B switch: {} vs {}, starting with {}", label_a, label_b, Label());
        return true;
    }

    [[nodiscard]] static bool IsRegistered() {
        return a.load() != nullptr;
    }

    [[nodiscard]] static bool IsB() {
        return is_b.load(std::memory_order_relaxed);
    }

    /// Name of the active variant, or nullptr without an experiment.
    [[nodiscard]] static const char* Label() {
        if (!IsRegistered()) {
            return nullptr;
        }
        return LabelOf(IsB());
    }

    /// Name of variant B (or A), or nullptr without an experiment.
    [[nodiscard]] static const char* LabelOf(bool variant_b) {
        return variant_b ? b.load() : a.load();
    }

    /// Counts switches, so a consumer can tell a window of measurements spans one.
    [[nodiscard]] static u64 Switches() {
        return switches.load(std::memory_order_acquire);
    }

    /// Callable from any thread (hotkey, launcher settings, the auto switch).
    static void Toggle() {
        if (!IsRegistered()) {
            return;
        }
        bool current = is_b.load();
        while (!is_b.compare_exchange_weak(current, !current)) {
        }
        Switched(!current);
    }

    /// Selects a variant; counts as a switch only if it changes.
    static void Set(bool variant_b) {
        if (!IsRegistered() || is_b.exchange(variant_b) == variant_b) {
            return;
        }
        Switched(variant_b);
    }

private:
    static void Switched(bool variant_b) {
        switches.fetch_add(1, std::memory_order_release);
        LOG_INFO(Common, "A/B switch: now {}", LabelOf(variant_b));
    }

    static inline std::atomic<const char*> a{nullptr};
    static inline std::atomic<const char*> b{nullptr};
    static inline std::atomic<bool> is_b{false};
    static inline std::atomic<u64> switches{0};
};

} // namespace Common
