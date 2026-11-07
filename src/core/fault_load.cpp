// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/fault_load.h"

#if defined(__linux__) && defined(__x86_64__)
#include <Zydis/Zydis.h>
#include <sys/ucontext.h>
#endif

namespace Core {

#if defined(__linux__) && defined(__x86_64__)

namespace {

/// The kernel's signal frame keeps the FXSAVE area, then (when this magic is in its software
/// bytes) the XSAVE header and the extended state in standard format.
constexpr size_t SwBytesOffset = 464;
constexpr u32 XstateMagic1 = 0x46505853;
constexpr size_t XstateHeaderOffset = 512;
constexpr size_t YmmHiOffset = 576;
constexpr u64 XfeatureSse = 1ULL << 1;
constexpr u64 XfeatureYmm = 1ULL << 2;

struct SwBytes {
    u32 magic1;
    u32 extended_size;
    u64 xfeatures;
    u32 xstate_size;
    u32 padding[7];
};

/// The signal frame's gregs index of each general-purpose register by Zydis id (rax, rcx, rdx,
/// rbx, rsp, rbp, rsi, rdi, r8-r15).
constexpr int GregIndex[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP,
                               REG_RSI, REG_RDI, REG_R8,  REG_R9,  REG_R10, REG_R11,
                               REG_R12, REG_R13, REG_R14, REG_R15};

const ZydisDecoder& Decoder() {
    static const ZydisDecoder decoder = [] {
        ZydisDecoder d;
        ZydisDecoderInit(&d, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
        return d;
    }();
    return decoder;
}

u8* FpState(ucontext_t* uc) {
    return reinterpret_cast<u8*>(uc->uc_mcontext.fpregs);
}

/// The XSAVE header's XSTATE_BV, if the frame has extended state with all of `features`.
u64* XstateBv(ucontext_t* uc, u64 features) {
    u8* fp = FpState(uc);
    if (fp == nullptr) {
        return nullptr;
    }
    SwBytes sw;
    std::memcpy(&sw, fp + SwBytesOffset, sizeof(sw));
    if (sw.magic1 != XstateMagic1 || (sw.xfeatures & features) != features ||
        sw.xstate_size < YmmHiOffset + 16 * 16) {
        return nullptr;
    }
    return reinterpret_cast<u64*>(fp + XstateHeaderOffset);
}

enum class Kind : u8 {
    None,
    Gpr,
    GprZero,
    GprSign,
    Vector,
};

Kind KindOf(ZydisMnemonic mnemonic) {
    switch (mnemonic) {
    case ZYDIS_MNEMONIC_MOV:
        return Kind::Gpr;
    case ZYDIS_MNEMONIC_MOVZX:
        return Kind::GprZero;
    case ZYDIS_MNEMONIC_MOVSX:
    case ZYDIS_MNEMONIC_MOVSXD:
        return Kind::GprSign;
    case ZYDIS_MNEMONIC_MOVSS:
    case ZYDIS_MNEMONIC_MOVSD:
    case ZYDIS_MNEMONIC_MOVD:
    case ZYDIS_MNEMONIC_MOVQ:
    case ZYDIS_MNEMONIC_MOVUPS:
    case ZYDIS_MNEMONIC_MOVAPS:
    case ZYDIS_MNEMONIC_MOVUPD:
    case ZYDIS_MNEMONIC_MOVAPD:
    case ZYDIS_MNEMONIC_MOVDQU:
    case ZYDIS_MNEMONIC_MOVDQA:
    case ZYDIS_MNEMONIC_LDDQU:
    case ZYDIS_MNEMONIC_MOVNTDQA:
    case ZYDIS_MNEMONIC_VMOVSS:
    case ZYDIS_MNEMONIC_VMOVSD:
    case ZYDIS_MNEMONIC_VMOVD:
    case ZYDIS_MNEMONIC_VMOVQ:
    case ZYDIS_MNEMONIC_VMOVUPS:
    case ZYDIS_MNEMONIC_VMOVAPS:
    case ZYDIS_MNEMONIC_VMOVUPD:
    case ZYDIS_MNEMONIC_VMOVAPD:
    case ZYDIS_MNEMONIC_VMOVDQU:
    case ZYDIS_MNEMONIC_VMOVDQA:
    case ZYDIS_MNEMONIC_VLDDQU:
    case ZYDIS_MNEMONIC_VMOVNTDQA:
        return Kind::Vector;
    default:
        return Kind::None;
    }
}

} // namespace

bool DecodeFaultLoad(void* context, FaultLoad& load) {
    auto* uc = static_cast<ucontext_t*>(context);
    auto& gregs = uc->uc_mcontext.gregs;
    const u64 rip = static_cast<u64>(gregs[REG_RIP]);
    ZydisDecodedInstruction instr;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&Decoder(), reinterpret_cast<const void*>(rip),
                                             ZYDIS_MAX_INSTRUCTION_LENGTH, &instr, operands))) {
        return false;
    }
    load.mnemonic = static_cast<u16>(instr.mnemonic);
    const Kind kind = KindOf(instr.mnemonic);
    if (kind == Kind::None || instr.operand_count_visible != 2) {
        return false;
    }
    const auto& dst = operands[0];
    const auto& src = operands[1];
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER || src.type != ZYDIS_OPERAND_TYPE_MEMORY ||
        src.mem.type != ZYDIS_MEMOP_TYPE_MEM || src.mem.segment == ZYDIS_REGISTER_FS ||
        src.mem.segment == ZYDIS_REGISTER_GS || src.size == 0 || src.size % 8 != 0) {
        return false;
    }
    load.size = src.size / 8;
    load.length = instr.length;
    load.vex = instr.encoding == ZYDIS_INSTRUCTION_ENCODING_VEX;
    if (instr.encoding != ZYDIS_INSTRUCTION_ENCODING_LEGACY && !load.vex) {
        return false;
    }

    const ZydisRegister reg = dst.reg.value;
    const ZydisRegisterClass reg_class = ZydisRegisterGetClass(reg);
    // Byte registers number ah-bh and spl-dil apart: number a general-purpose register by the
    // 64-bit one it's part of.
    const bool is_gpr = reg_class == ZYDIS_REGCLASS_GPR8 || reg_class == ZYDIS_REGCLASS_GPR16 ||
                        reg_class == ZYDIS_REGCLASS_GPR32 || reg_class == ZYDIS_REGCLASS_GPR64;
    const ZyanI8 id = ZydisRegisterGetId(
        is_gpr ? ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg) : reg);
    if (id < 0 || id > 15) {
        return false;
    }
    load.reg = static_cast<u8>(id);
    if (kind == Kind::Vector) {
        if (reg_class == ZYDIS_REGCLASS_XMM) {
            load.dest = FaultLoad::Dest::Xmm;
        } else if (reg_class == ZYDIS_REGCLASS_YMM && load.vex) {
            load.dest = FaultLoad::Dest::Ymm;
        } else {
            return false; // MMX registers, and anything wider
        }
        const u32 width = load.dest == FaultLoad::Dest::Ymm ? 32 : 16;
        if (load.size > width) {
            return false;
        }
        const u64 needed = load.vex ? (XfeatureSse | XfeatureYmm) : 0;
        if (needed != 0 && XstateBv(uc, needed) == nullptr) {
            return false;
        }
        if (FpState(uc) == nullptr) {
            return false;
        }
    } else {
        switch (reg_class) {
        case ZYDIS_REGCLASS_GPR64:
            load.reg_size = 8;
            break;
        case ZYDIS_REGCLASS_GPR32:
            load.reg_size = 4;
            break;
        case ZYDIS_REGCLASS_GPR16:
            load.reg_size = 2;
            break;
        case ZYDIS_REGCLASS_GPR8:
            // ah, ch, dh and bh live in bits 8-15 of their register.
            if (reg == ZYDIS_REGISTER_AH || reg == ZYDIS_REGISTER_CH || reg == ZYDIS_REGISTER_DH ||
                reg == ZYDIS_REGISTER_BH) {
                return false;
            }
            load.reg_size = 1;
            break;
        default:
            return false;
        }
        if (load.size > load.reg_size) {
            return false;
        }
        load.dest = FaultLoad::Dest::Gpr;
        load.extend = kind == Kind::GprZero   ? FaultLoad::Extend::Zero
                      : kind == Kind::GprSign ? FaultLoad::Extend::Sign
                                              : FaultLoad::Extend::None;
    }

    ZydisRegisterContext regs{};
    for (int i = 0; i < 16; ++i) {
        const u64 value = static_cast<u64>(gregs[GregIndex[i]]);
        regs.values[ZYDIS_REGISTER_RAX + i] = value;
        regs.values[ZYDIS_REGISTER_EAX + i] = value & 0xffffffffULL;
    }
    regs.values[ZYDIS_REGISTER_RIP] = rip;
    ZyanU64 address = 0;
    if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddressEx(&instr, &src, rip, &regs, &address))) {
        return false;
    }
    load.address = static_cast<VAddr>(address);
    return true;
}

bool CompleteFaultLoad(void* context, const FaultLoad& load, const u8* data) {
    auto* uc = static_cast<ucontext_t*>(context);
    auto& gregs = uc->uc_mcontext.gregs;
    if (load.dest == FaultLoad::Dest::Gpr) {
        u64 value = 0;
        std::memcpy(&value, data, load.size);
        if (load.extend == FaultLoad::Extend::Sign && load.size < 8) {
            const u32 shift = 64 - load.size * 8;
            value = static_cast<u64>(static_cast<s64>(value << shift) >> shift);
        }
        auto& reg = gregs[GregIndex[load.reg]];
        if (load.reg_size >= 4) {
            // Writing a 32-bit register clears the upper half of its 64-bit one.
            reg = static_cast<greg_t>(load.reg_size == 4 ? (value & 0xffffffffULL) : value);
        } else {
            const u64 mask = (1ULL << (load.reg_size * 8)) - 1;
            reg = static_cast<greg_t>((static_cast<u64>(reg) & ~mask) | (value & mask));
        }
    } else {
        u8* fp = FpState(uc);
        if (fp == nullptr) {
            return false;
        }
        if (u64* bv = XstateBv(uc, XfeatureSse)) {
            // Components in their initial state aren't restored from memory, which may hold
            // anything for them; their initial state is all zeros.
            if ((*bv & XfeatureSse) == 0) {
                for (int i = 0; i < 16; ++i) {
                    std::memset(uc->uc_mcontext.fpregs->_xmm[i].element, 0, 16);
                }
                *bv |= XfeatureSse;
            }
        }
        u8 low[16]{};
        std::memcpy(low, data, std::min<u32>(load.size, 16));
        std::memcpy(uc->uc_mcontext.fpregs->_xmm[load.reg].element, low, 16);
        if (load.vex) {
            u64* bv = XstateBv(uc, XfeatureSse | XfeatureYmm);
            if (bv == nullptr) {
                return false;
            }
            u8* ymm_hi = fp + YmmHiOffset;
            if ((*bv & XfeatureYmm) == 0) {
                std::memset(ymm_hi, 0, 16 * 16);
                *bv |= XfeatureYmm;
            }
            u8 high[16]{};
            if (load.size > 16) {
                std::memcpy(high, data + 16, load.size - 16);
            }
            std::memcpy(ymm_hi + 16 * load.reg, high, 16);
        }
    }
    gregs[REG_RIP] += load.length;
    return true;
}

#else

bool DecodeFaultLoad(void*, FaultLoad&) {
    return false;
}

bool CompleteFaultLoad(void*, const FaultLoad&, const u8*) {
    return false;
}

#endif

} // namespace Core
