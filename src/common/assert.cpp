// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include "common/arch.h"
#include "common/assert.h"
#include "common/soft_assert.h"
#include "core/signals.h"
#include "emulator.h"
#include "launcher/session.h"

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

namespace Common {

namespace {
thread_local int soft_asserts = 0;
}

SoftAsserts::SoftAsserts() {
    ++soft_asserts;
}

SoftAsserts::~SoftAsserts() {
    --soft_asserts;
}

bool SoftAsserts::Active() {
    return soft_asserts > 0;
}

void SoftAsserts::EndOnThisThread() {
    soft_asserts = std::numeric_limits<int>::min() / 2;
}

} // namespace Common

bool assert_is_soft() {
    return Common::SoftAsserts::Active();
}

void assert_fail_impl() {
    if (Common::SoftAsserts::Active()) {
        throw Common::SoftAssertFailure{};
    }
    Launcher::Session::Failed("shadPS4 stopped on an internal error; its log has the details.",
                              "fatal_error");
    Core::Signals::Instance()->RemoveHandlers();
    Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    Crash();
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    if (Common::SoftAsserts::Active()) {
        LOG_WARNING(Debug, "Assertion Failed!\n{}", msg);
        throw Common::SoftAssertFailure{};
    }
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    Launcher::Session::Failed(msg, "fatal_error");
    assert_fail_impl();
}
