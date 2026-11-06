// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "shader_recompiler/ir/ir_emitter.h"

namespace Shader {

/// Converts an element read as raw dwords by its RuntimeBufferFormat (`format`, a uvec4 from
/// the stage table), without branches: each component is extracted by its offset and width,
/// moved up by `byte_shift` bits, converted every way its kind could need, and the kind picks
/// one; then the destination select. Returns four floats, or their bits for integer kinds.
IR::Value DecodeRuntimeFormat(IR::IREmitter& ir, const std::array<IR::U32, 4>& dwords,
                              const IR::Value& format, const IR::U32& byte_shift);

} // namespace Shader
