// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core {

/// A plain load that faulted on a protected page, decoded so it can be finished without the
/// page: the fault handler reads the bytes itself and writes them to the destination register.
/// Only moves into a general-purpose or vector register are decoded, as those are all a load
/// that only reads memory can be.
struct FaultLoad {
    enum class Dest : u8 {
        Gpr,
        Xmm,
        Ymm,
    };
    enum class Extend : u8 {
        /// Merges into the low bytes of an 8 or 16-bit destination, zero-extends a 32-bit one.
        None,
        Zero,
        Sign,
    };

    /// Address and size of the bytes the instruction reads.
    VAddr address{};
    u32 size{};
    u32 length{};
    Dest dest{};
    Extend extend{};
    /// The destination register's number (0-15) and, for a general-purpose one, its size.
    u8 reg{};
    u8 reg_size{};
    /// A VEX-encoded vector load, which zeroes the destination above what it loads up to 256
    /// bits. A legacy SSE load leaves the upper half of the YMM register as it is.
    bool vex{};
    /// The Zydis mnemonic, for counting the loads that couldn't be decoded.
    u16 mnemonic{};
};

/// Decodes the instruction at the signal context's instruction pointer as a FaultLoad. Returns
/// false for anything else (with `load.mnemonic` set if it decoded at all), on platforms other
/// than x86-64 Linux, and where the context lacks the state it would need.
bool DecodeFaultLoad(void* context, FaultLoad& load);

/// Writes `data` (load.size bytes) to the load's destination register as the instruction would
/// have and steps the context past it.
bool CompleteFaultLoad(void* context, const FaultLoad& load, const u8* data);

} // namespace Core
