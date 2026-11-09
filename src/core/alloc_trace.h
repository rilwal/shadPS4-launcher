// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>
#include "common/types.h"

namespace Core::AllocTrace {

/// With SHADPS4_ALLOC_TRACE=1: for a libc allocation function a module imports (malloc, free,
/// new, delete and their kin), returns a host wrapper that calls `original` and keeps live
/// bytes per call site, logging the largest and fastest-growing sites every 5 s. Otherwise
/// returns `original`. Diagnostic only.
u64 Interpose(std::string_view name, u64 original);

} // namespace Core::AllocTrace
