// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reinterpret.h"
#include "shader_recompiler/ir/runtime_format_decode.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_buffer_format.h"
#include "video_core/amdgpu/resource.h"

namespace Shader::Optimization {

struct FormatInfo {
    AmdGpu::DataFormat data_format;
    AmdGpu::NumberFormat num_format;
    AmdGpu::CompMapping swizzle;
    AmdGpu::NumberConversion num_conversion;
    u32 num_components;
};

static bool IsBufferFormatLoad(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadBufferFormatF32;
}

static bool IsBufferFormatStore(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::StoreBufferFormatF32;
}

static IR::Value LoadBufferFormat(IR::IREmitter& ir, const IR::Value handle, const IR::U32 address,
                                  const IR::BufferInstInfo info, const FormatInfo& format_info) {
    IR::Value interpreted;
    switch (format_info.data_format) {
    case AmdGpu::DataFormat::FormatInvalid:
        interpreted = ir.Imm32(0.f);
        break;
    case AmdGpu::DataFormat::Format8: {
        const auto raw = ir.UConvert(32, ir.LoadBufferU8(handle, address, info));
        const auto unpacked = ir.Unpack4x8(format_info.num_format, raw);
        interpreted = ir.CompositeExtract(unpacked, 0);
        break;
    }
    case AmdGpu::DataFormat::Format8_8: {
        const auto raw = ir.UConvert(32, ir.LoadBufferU16(handle, address, info));
        const auto unpacked = ir.Unpack4x8(format_info.num_format, raw);
        interpreted = ir.CompositeConstruct(ir.CompositeExtract(unpacked, 0),
                                            ir.CompositeExtract(unpacked, 1));
        break;
    }
    case AmdGpu::DataFormat::Format8_8_8_8:
        interpreted = ir.Unpack4x8(format_info.num_format,
                                   IR::U32{ir.LoadBufferU32(1, handle, address, info)});
        break;
    case AmdGpu::DataFormat::Format16: {
        const auto raw = ir.UConvert(32, ir.LoadBufferU16(handle, address, info));
        const auto unpacked = ir.Unpack2x16(format_info.num_format, raw);
        interpreted = ir.CompositeExtract(unpacked, 0);
        break;
    }
    case AmdGpu::DataFormat::Format16_16:
        interpreted = ir.Unpack2x16(format_info.num_format,
                                    IR::U32{ir.LoadBufferU32(1, handle, address, info)});
        break;
    case AmdGpu::DataFormat::Format10_11_11:
        interpreted = ir.Unpack10_11_11(format_info.num_format,
                                        IR::U32{ir.LoadBufferU32(1, handle, address, info)});
        break;
    case AmdGpu::DataFormat::Format2_10_10_10:
        interpreted = ir.Unpack2_10_10_10(format_info.num_format,
                                          IR::U32{ir.LoadBufferU32(1, handle, address, info)});
        break;
    case AmdGpu::DataFormat::Format16_16_16_16: {
        const auto raw = ir.LoadBufferU32(2, handle, address, info);
        interpreted = ir.CompositeConstruct(
            ir.Unpack2x16(format_info.num_format, IR::U32{ir.CompositeExtract(raw, 0)}),
            ir.Unpack2x16(format_info.num_format, IR::U32{ir.CompositeExtract(raw, 1)}));
        break;
    }
    case AmdGpu::DataFormat::Format32:
    case AmdGpu::DataFormat::Format32_32:
    case AmdGpu::DataFormat::Format32_32_32:
    case AmdGpu::DataFormat::Format32_32_32_32: {
        ASSERT(format_info.num_format == AmdGpu::NumberFormat::Uint ||
               format_info.num_format == AmdGpu::NumberFormat::Sint ||
               format_info.num_format == AmdGpu::NumberFormat::Float);
        interpreted = ir.LoadBufferF32(format_info.num_components, handle, address, info);
        break;
    }
    default:
        UNREACHABLE_MSG("Unsupported buffer data format: {}", format_info.data_format);
    }

    // Pad to 4 components and apply additional modifications.
    boost::container::static_vector<IR::Value, 4> components;
    for (u32 i = 0; i < 4; i++) {
        if (i < format_info.num_components) {
            const auto component =
                IR::F32{format_info.num_components == 1 ? interpreted
                                                        : ir.CompositeExtract(interpreted, i)};
            components.push_back(
                ApplyReadNumberConversion(ir, component, format_info.num_conversion));
        } else {
            components.push_back(ir.Imm32(0.f));
        }
    }
    const auto swizzled = ApplySwizzle(ir, ir.CompositeConstruct(components), format_info.swizzle);
    return swizzled;
}

/// Reads a formatted element whose format is in the stage table (see RuntimeBufferFormat).
static IR::Value LoadRuntimeBufferFormat(IR::IREmitter& ir, const IR::Value handle,
                                         const IR::U32 address, const IR::BufferInstInfo info) {
    const IR::Value format = ir.BufferFormat(handle);
    // The address is in bytes. Aligned like the specialised path's loads, no component crosses
    // a dword, and four dwords from the element's dword cover every format.
    const IR::U32 address_mask{ir.CompositeExtract(format, 3)};
    const IR::U32 aligned{ir.BitwiseAnd(address, ir.BitwiseNot(address_mask))};
    const IR::U32 dword_address{ir.ShiftRightLogical(aligned, ir.Imm32(2u))};
    const IR::U32 byte_shift{
        ir.ShiftLeftLogical(ir.BitwiseAnd(aligned, ir.Imm32(3u)), ir.Imm32(3u))};
    const IR::Value raw = ir.LoadBufferU32(4, handle, dword_address, info);
    const std::array<IR::U32, 4> dwords{
        IR::U32{ir.CompositeExtract(raw, 0)}, IR::U32{ir.CompositeExtract(raw, 1)},
        IR::U32{ir.CompositeExtract(raw, 2)}, IR::U32{ir.CompositeExtract(raw, 3)}};
    return DecodeRuntimeFormat(ir, dwords, format, byte_shift);
}

} // namespace Shader::Optimization

namespace Shader {

IR::Value DecodeRuntimeFormat(IR::IREmitter& ir, const std::array<IR::U32, 4>& dwords,
                              const IR::Value& format, const IR::U32& byte_shift) {
    using Kind = RuntimeBufferFormat::Kind;
    const IR::U32 offsets{ir.CompositeExtract(format, 0)};
    const IR::U32 widths{ir.CompositeExtract(format, 1)};
    const IR::U32 kind_swizzle{ir.CompositeExtract(format, 2)};
    const IR::U32 kind{ir.BitFieldExtract(kind_swizzle, ir.Imm32(0u), ir.Imm32(4u))};
    const auto is_kind = [&](Kind k) { return ir.IEqual(kind, ir.Imm32(static_cast<u32>(k))); };

    const IR::F32 zero = ir.Imm32(0.f);
    const IR::F32 one = ir.Imm32(1.f);
    std::array<IR::F32, 4> components;
    for (u32 i = 0; i < 4; ++i) {
        const IR::U32 width{ir.BitFieldExtract(widths, ir.Imm32(i * 8), ir.Imm32(8u))};
        const IR::U32 bit{
            ir.IAdd(byte_shift, ir.BitFieldExtract(offsets, ir.Imm32(i * 8), ir.Imm32(8u)))};
        const IR::U32 dword_index{ir.ShiftRightLogical(bit, ir.Imm32(5u))};
        const IR::U32 dword{ir.Select(
            ir.IEqual(dword_index, ir.Imm32(0u)), dwords[0],
            ir.Select(ir.IEqual(dword_index, ir.Imm32(1u)), dwords[1],
                      ir.Select(ir.IEqual(dword_index, ir.Imm32(2u)), dwords[2], dwords[3])))};
        const IR::U32 bit_in_dword{ir.BitwiseAnd(bit, ir.Imm32(31u))};
        const IR::U32 u{ir.BitFieldExtract(dword, bit_in_dword, width)};
        const IR::U32 s{ir.BitFieldExtract(dword, bit_in_dword, width, true)};

        // Normalising divisors, kept in range where the width can't be normalised.
        const IR::U32 norm_width{ir.UMin(ir.UMax(width, ir.Imm32(1u)), ir.Imm32(31u))};
        const IR::F32 unorm_max{ir.ConvertUToF(
            32, 32, ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1u), norm_width), ir.Imm32(1u)))};
        const IR::F32 snorm_max{ir.ConvertUToF(
            32, 32,
            ir.UMax(ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1u), ir.ISub(norm_width, ir.Imm32(1u))),
                            ir.Imm32(1u)),
                    ir.Imm32(1u)))};
        const IR::F32 as_uint{ir.ConvertUToF(32, 32, u)};
        const IR::F32 as_sint{ir.ConvertSToF(32, 32, s)};
        const IR::F32 unorm{ir.FPDiv(as_uint, unorm_max)};
        const IR::F32 snorm{ir.FPMax(ir.FPDiv(as_sint, snorm_max), ir.Imm32(-1.f))};
        const IR::F32 half{ir.CompositeExtract(ir.Unpack2x16(AmdGpu::NumberFormat::Float, u), 0)};

        IR::F32 value{ir.BitCast<IR::F32, IR::U32>(u)};
        value = IR::F32{ir.Select(is_kind(Kind::Sint), ir.BitCast<IR::F32, IR::U32>(s), value)};
        value = IR::F32{ir.Select(is_kind(Kind::Unorm), unorm, value)};
        value = IR::F32{ir.Select(is_kind(Kind::Snorm), snorm, value)};
        value = IR::F32{ir.Select(is_kind(Kind::Half), half, value)};
        value = IR::F32{ir.Select(is_kind(Kind::Uscaled), as_uint, value)};
        value = IR::F32{ir.Select(is_kind(Kind::Sscaled), as_sint, value)};
        // Components the format doesn't have read as zero before the swizzle, and a vertex
        // attribute's fourth as one.
        IR::F32 missing = zero;
        if (i == 3) {
            const auto bit = [&](u32 mask) {
                return ir.INotEqual(ir.BitwiseAnd(kind_swizzle, ir.Imm32(mask)), ir.Imm32(0u));
            };
            const IR::F32 one_value{ir.Select(bit(RuntimeBufferFormat::FillAlphaIntegerBit),
                                              ir.BitCast<IR::F32, IR::U32>(ir.Imm32(1u)), one)};
            missing =
                IR::F32{ir.Select(bit(RuntimeBufferFormat::FillAlphaOneBit), one_value, zero)};
        }
        components[i] = IR::F32{ir.Select(ir.IEqual(width, ir.Imm32(0u)), missing, value)};
    }

    std::array<IR::Value, 4> swizzled;
    for (u32 i = 0; i < 4; ++i) {
        const IR::U32 select{ir.BitFieldExtract(
            kind_swizzle, ir.Imm32(RuntimeBufferFormat::SwizzleShift + i * 3), ir.Imm32(3u))};
        const auto is = [&](AmdGpu::CompSwizzle c) {
            return ir.IEqual(select, ir.Imm32(static_cast<u32>(c)));
        };
        IR::Value value = ir.Select(is(AmdGpu::CompSwizzle::One), one, zero);
        value = ir.Select(is(AmdGpu::CompSwizzle::Red), components[0], value);
        value = ir.Select(is(AmdGpu::CompSwizzle::Green), components[1], value);
        value = ir.Select(is(AmdGpu::CompSwizzle::Blue), components[2], value);
        value = ir.Select(is(AmdGpu::CompSwizzle::Alpha), components[3], value);
        swizzled[i] = value;
    }
    return ir.CompositeConstruct(swizzled[0], swizzled[1], swizzled[2], swizzled[3]);
}

} // namespace Shader

namespace Shader::Optimization {

static void StoreBufferFormat(IR::IREmitter& ir, const IR::Value handle, const IR::U32 address,
                              const IR::Value& value, const IR::BufferInstInfo info,
                              const FormatInfo& format_info) {
    // Extract actual number of components and apply additional modifications.
    const auto swizzled = ApplySwizzle(ir, value, format_info.swizzle.Inverse());
    boost::container::static_vector<IR::Value, 4> components;
    for (u32 i = 0; i < format_info.num_components; i++) {
        const auto component = IR::F32{ir.CompositeExtract(swizzled, i)};
        components.push_back(ApplyWriteNumberConversion(ir, component, format_info.num_conversion));
    }
    const auto real_value =
        components.size() == 1 ? components[0] : ir.CompositeConstruct(components);

    switch (format_info.data_format) {
    case AmdGpu::DataFormat::FormatInvalid:
        break;
    case AmdGpu::DataFormat::Format8: {
        const auto packed =
            ir.Pack4x8(format_info.num_format, ir.CompositeConstruct(real_value, ir.Imm32(0.f),
                                                                     ir.Imm32(0.f), ir.Imm32(0.f)));
        ir.StoreBufferU8(handle, address, ir.UConvert(8, packed), info);
        break;
    }
    case AmdGpu::DataFormat::Format8_8: {
        const auto packed = ir.Pack4x8(format_info.num_format,
                                       ir.CompositeConstruct(ir.CompositeExtract(real_value, 0),
                                                             ir.CompositeExtract(real_value, 1),
                                                             ir.Imm32(0.f), ir.Imm32(0.f)));
        ir.StoreBufferU16(handle, address, ir.UConvert(16, packed), info);
        break;
    }
    case AmdGpu::DataFormat::Format8_8_8_8: {
        auto packed = ir.Pack4x8(format_info.num_format, real_value);
        ir.StoreBufferU32(1, handle, address, packed, info);
        break;
    }
    case AmdGpu::DataFormat::Format16: {
        const auto packed =
            ir.Pack2x16(format_info.num_format, ir.CompositeConstruct(real_value, ir.Imm32(0.f)));
        ir.StoreBufferU16(handle, address, ir.UConvert(16, packed), info);
        break;
    }
    case AmdGpu::DataFormat::Format16_16: {
        const auto packed = ir.Pack2x16(format_info.num_format, real_value);
        ir.StoreBufferU32(1, handle, address, packed, info);
        break;
    }
    case AmdGpu::DataFormat::Format10_11_11: {
        const auto packed = ir.Pack10_11_11(format_info.num_format, real_value);
        ir.StoreBufferU32(1, handle, address, packed, info);
        break;
    }
    case AmdGpu::DataFormat::Format2_10_10_10: {
        const auto packed = ir.Pack2_10_10_10(format_info.num_format, real_value);
        ir.StoreBufferU32(1, handle, address, packed, info);
        break;
    }
    case AmdGpu::DataFormat::Format16_16_16_16: {
        const auto packed = ir.CompositeConstruct(
            ir.Pack2x16(format_info.num_format,
                        ir.CompositeConstruct(ir.CompositeExtract(real_value, 0),
                                              ir.CompositeExtract(real_value, 1))),
            ir.Pack2x16(format_info.num_format,
                        ir.CompositeConstruct(ir.CompositeExtract(real_value, 2),
                                              ir.CompositeExtract(real_value, 3))));
        ir.StoreBufferU32(2, handle, address, packed, info);
        break;
    }
    case AmdGpu::DataFormat::Format32:
    case AmdGpu::DataFormat::Format32_32:
    case AmdGpu::DataFormat::Format32_32_32:
    case AmdGpu::DataFormat::Format32_32_32_32: {
        ASSERT(format_info.num_format == AmdGpu::NumberFormat::Uint ||
               format_info.num_format == AmdGpu::NumberFormat::Sint ||
               format_info.num_format == AmdGpu::NumberFormat::Float);
        ir.StoreBufferF32(format_info.num_components, handle, address, real_value, info);
        break;
    }
    default:
        UNREACHABLE_MSG("Unsupported buffer data format: {}", format_info.data_format);
    }
}

static void LowerBufferFormatInst(IR::Block& block, IR::Inst& inst, Info& info,
                                  const Profile& profile) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    const auto desc{info.buffers[inst.Arg(0).U32()]};
    const auto buffer{desc.GetSharp(info)};

    const auto is_inst_typed = flags.inst_data_fmt != AmdGpu::DataFormat::FormatInvalid;
    if (!is_inst_typed && IsBufferFormatLoad(inst) &&
        UsesRuntimeBufferFormat(profile.runtime_buffer_formats, desc, buffer)) {
        inst.ReplaceUsesWithAndRemove(
            LoadRuntimeBufferFormat(ir, inst.Arg(0), IR::U32{inst.Arg(1)}, flags));
        return;
    }
    const auto data_format =
        is_inst_typed ? AmdGpu::RemapDataFormat(flags.inst_data_fmt.Value()) : buffer.GetDataFmt();
    const auto format_info = FormatInfo{
        .data_format = data_format,
        .num_format = is_inst_typed
                          ? AmdGpu::RemapNumberFormat(flags.inst_num_fmt.Value(), data_format)
                          : buffer.GetNumberFmt(),
        .swizzle = is_inst_typed
                       ? AmdGpu::RemapSwizzle(flags.inst_data_fmt.Value(), AmdGpu::IdentityMapping)
                       : buffer.DstSelect(),
        .num_conversion = is_inst_typed ? AmdGpu::MapNumberConversion(flags.inst_num_fmt.Value(),
                                                                      flags.inst_data_fmt.Value())
                                        : buffer.GetNumberConversion(),
        .num_components = AmdGpu::NumComponents(data_format),
    };

    if (data_format == AmdGpu::DataFormat::FormatInvalid) {
        LOG_WARNING(Render_Recompiler, "Skipping lowering for null buffer sharp");
        if (IsBufferFormatLoad(inst)) {
            inst.ReplaceUsesWithAndRemove(
                ir.CompositeConstruct(ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f), ir.Imm32(0.f)));
        } else {
            inst.Invalidate();
        }
        return;
    }

    if (IsBufferFormatLoad(inst)) {
        const auto interpreted =
            LoadBufferFormat(ir, inst.Arg(0), IR::U32{inst.Arg(1)}, flags, format_info);
        inst.ReplaceUsesWithAndRemove(interpreted);
    } else if (IsBufferFormatStore(inst)) {
        StoreBufferFormat(ir, inst.Arg(0), IR::U32{inst.Arg(1)}, inst.Arg(2), flags, format_info);
        inst.Invalidate();
    }
}

void LowerBufferFormatToRaw(IR::Program& program, const Profile& profile) {
    auto& info = program.info;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (IsBufferFormatLoad(inst) || IsBufferFormatStore(inst)) {
                LowerBufferFormatInst(*block, inst, info, profile);
            }
        }
    }
}

} // namespace Shader::Optimization
