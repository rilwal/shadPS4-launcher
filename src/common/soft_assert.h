// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <stdexcept>

namespace Common {

/// Thrown by a failed ASSERT or UNREACHABLE on a thread inside a SoftAsserts scope.
struct SoftAssertFailure : std::runtime_error {
    SoftAssertFailure() : std::runtime_error{"assertion failed"} {}
};

/// While one is alive on a thread, a failed ASSERT or UNREACHABLE there throws
/// SoftAssertFailure instead of stopping the emulator. For work on guesses that may be wrong,
/// such as translating a shader before any draw says what it is, where the caller drops the
/// result instead.
class SoftAsserts {
public:
    SoftAsserts();
    ~SoftAsserts();
    SoftAsserts(const SoftAsserts&) = delete;
    SoftAsserts& operator=(const SoftAsserts&) = delete;

    /// Whether the calling thread is inside a SoftAsserts scope.
    static bool Active();
    /// Makes failures on the calling thread fatal again, for good: for signal handlers, where
    /// throwing would unwind through a frame the signal interrupted.
    static void EndOnThisThread();
};

} // namespace Common
