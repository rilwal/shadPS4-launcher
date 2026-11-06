// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// What the shader recompiler needs from the rest of the emulator, for the shader tool: logging
// to stderr, asserts that throw under SoftAsserts and abort otherwise, and default settings.

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <fmt/format.h>

#include "common/assert.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "common/soft_assert.h"
#include "core/emulator_settings.h"
#include "core/signals.h"

namespace Common::Log {

std::array<Level, NUM_LOG_CLASSES> g_class_levels = [] {
    std::array<Level, NUM_LOG_CLASSES> levels{};
    levels.fill(std::getenv("SHADER_TOOL_LOG") ? Level::Info : Level::Error);
    return levels;
}();

void VLog(Class, Level level, const char* file, int line, const char* func, fmt::string_view format,
          fmt::format_args args) {
    if (level < Level::Error && !std::getenv("SHADER_TOOL_LOG")) {
        return;
    }
    fmt::print(stderr, "{}:{} {}: {}\n", file, line, func, fmt::vformat(format, args));
}

} // namespace Common::Log

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
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    if (Common::SoftAsserts::Active()) {
        throw Common::SoftAssertFailure{};
    }
    std::fprintf(stderr, "Assertion failed: %s\n", msg);
    std::abort();
}

// Defaults, apart from what main sets from the saved host.
EmulatorSettingsImpl::EmulatorSettingsImpl() = default;
EmulatorSettingsImpl::~EmulatorSettingsImpl() = default;

std::shared_ptr<EmulatorSettingsImpl> EmulatorSettingsImpl::GetInstance() {
    static const auto instance = std::make_shared<EmulatorSettingsImpl>();
    return instance;
}

// The recompiler only needs these for shader dumps and the SRT walker's fault handler, which
// the tool never uses.
namespace Common::FS {

const std::filesystem::path& GetUserPath(PathType) {
    static const auto path = std::filesystem::temp_directory_path();
    return path;
}

IOFile::IOFile(const std::filesystem::path&, FileAccessMode, FileType, FileShareFlag) {
    std::fputs("shader_tool: file writes aren't supported\n", stderr);
    std::abort();
}

IOFile::~IOFile() = default;

} // namespace Common::FS

namespace Common {
void* GetRip(void*) {
    std::abort();
}
} // namespace Common

namespace Core {
SignalDispatch::SignalDispatch() = default;
SignalDispatch::~SignalDispatch() = default;
} // namespace Core
