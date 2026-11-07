// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core {

/// A plain load that faulted on a protected page, decoded so it can be finished without the
/// page: the fault handler reads the bytes itself and writes them to the destination register.
/// Moves into a general-purpose or vector register, cmp and test with a memory operand, and
/// movs (with or without rep) are decoded.
struct FaultLoad {
    enum class Op : u8 {
        /// A move into a register.
        Load,
        /// cmp or test with a memory operand: only sets flags.
        Compare,
        Test,
        /// movs, with or without rep: copies `size` bytes from `address` to `store_address`. The
        /// GPU thread does the copy, as the destination may be tracked memory too.
        Copy,
    };
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

    Op op{};
    /// Address and size of the bytes the instruction reads.
    VAddr address{};
    u32 size{};
    /// Copy: where the bytes go, and the element size it steps rsi, rdi and rcx by.
    VAddr store_address{};
    u8 element{};
    bool rep{};
    /// Compare and Test: the other operand, a register (`reg`, `reg_size`) or an immediate,
    /// and whether memory is the first operand (cmp mem, x) or the second (cmp reg, mem).
    bool other_is_reg{};
    bool memory_first{};
    u64 imm{};
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

/// Finishes the instruction from `data` (load.size bytes; unused for a Copy, which the caller
/// has done) as it would have run, and steps the context past it.
bool CompleteFaultLoad(void* context, const FaultLoad& load, const u8* data);

/// The faulting instruction's address, or 0 where it isn't known (or `context` is null).
u64 FaultInstructionAddress(void* context);

} // namespace Core
