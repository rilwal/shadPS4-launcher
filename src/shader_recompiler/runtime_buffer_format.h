// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <optional>

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/resource.h"

namespace Shader {

/// With runtime buffer formats, how a shader finds and converts each component of a formatted
/// buffer's element, as an entry in the stage table. One shader then reads buffers of any
/// format this can describe, instead of being compiled again for each.
struct RuntimeBufferFormat {
    enum class Kind : u32 {
        /// The bits as they are: Uint, and 32-bit Float.
        Raw = 0,
        /// Sign-extended bits.
        Sint = 1,
        Unorm = 2,
        Snorm = 3,
        /// 16-bit Float.
        Half = 4,
        Uscaled = 5,
        Sscaled = 6,
    };
    /// Bit offset of each component in the element, 8 bits each.
    u32 offsets{};
    /// Bit width of each component, 8 bits each; 0 for components the format doesn't have.
    u32 widths{};
    /// Kind in bits 0-3, then the destination select of x, y, z and w, 3 bits each.
    u32 kind_swizzle{};
    /// Low address bits to clear, as the specialised path reads elements at the granularity of
    /// its load (bytes, words or dwords) and so ignores them.
    u32 address_mask{};

    static constexpr u32 SwizzleShift = 4;
};
static_assert(sizeof(RuntimeBufferFormat) == 16);

/// Describes a buffer's format for reading at run time, if it can be. Null buffers describe as
/// all zero, which reads zeros like the null path does.
inline std::optional<RuntimeBufferFormat> DescribeRuntimeBufferFormat(
    const AmdGpu::Buffer& buffer) {
    using AmdGpu::DataFormat;
    using AmdGpu::NumberFormat;
    using Kind = RuntimeBufferFormat::Kind;
    const DataFormat data_format = buffer.GetDataFmt();
    if (data_format == DataFormat::FormatInvalid) {
        return RuntimeBufferFormat{};
    }
    std::array<u8, 4> widths{};
    u32 address_mask = 3;
    switch (data_format) {
    case DataFormat::Format8:
        widths = {8, 0, 0, 0};
        address_mask = 0;
        break;
    case DataFormat::Format16:
        widths = {16, 0, 0, 0};
        address_mask = 1;
        break;
    case DataFormat::Format8_8:
        widths = {8, 8, 0, 0};
        address_mask = 1;
        break;
    case DataFormat::Format32:
        widths = {32, 0, 0, 0};
        break;
    case DataFormat::Format16_16:
        widths = {16, 16, 0, 0};
        break;
    case DataFormat::Format2_10_10_10:
        widths = {10, 10, 10, 2};
        break;
    case DataFormat::Format8_8_8_8:
        widths = {8, 8, 8, 8};
        break;
    case DataFormat::Format32_32:
        widths = {32, 32, 0, 0};
        break;
    case DataFormat::Format16_16_16_16:
        widths = {16, 16, 16, 16};
        break;
    case DataFormat::Format32_32_32:
        widths = {32, 32, 32, 0};
        break;
    case DataFormat::Format32_32_32_32:
        widths = {32, 32, 32, 32};
        break;
    default:
        // Packed floats and the rest stay specialised.
        return std::nullopt;
    }
    const u32 width = widths[0];
    Kind kind;
    switch (buffer.GetNumberFmt()) {
    case NumberFormat::Uint:
        kind = Kind::Raw;
        break;
    case NumberFormat::Sint:
        kind = Kind::Sint;
        break;
    case NumberFormat::Unorm:
        kind = Kind::Unorm;
        break;
    case NumberFormat::Snorm:
        kind = Kind::Snorm;
        break;
    case NumberFormat::Float:
        if (width == 16) {
            kind = Kind::Half;
        } else if (width == 32) {
            kind = Kind::Raw;
        } else {
            return std::nullopt;
        }
        break;
    default:
        return std::nullopt;
    }
    if ((kind == Kind::Unorm || kind == Kind::Snorm) && width == 32) {
        return std::nullopt;
    }
    switch (buffer.GetNumberConversion()) {
    case AmdGpu::NumberConversion::None:
        break;
    case AmdGpu::NumberConversion::UintToUscaled:
        if (kind != Kind::Raw) {
            return std::nullopt;
        }
        kind = Kind::Uscaled;
        break;
    case AmdGpu::NumberConversion::SintToSscaled:
        if (kind != Kind::Sint) {
            return std::nullopt;
        }
        kind = Kind::Sscaled;
        break;
    default:
        return std::nullopt;
    }
    RuntimeBufferFormat format{.address_mask = address_mask};
    u32 offset = 0;
    for (u32 i = 0; i < 4; ++i) {
        format.offsets |= offset << (i * 8);
        format.widths |= u32(widths[i]) << (i * 8);
        offset += widths[i];
    }
    const auto dst_select = buffer.DstSelect();
    format.kind_swizzle = static_cast<u32>(kind);
    for (u32 i = 0; i < 4; ++i) {
        format.kind_swizzle |= (static_cast<u32>(dst_select.array[i]) & 7)
                               << (RuntimeBufferFormat::SwizzleShift + i * 3);
    }
    return format;
}

/// Size in bytes of an element a runtime format describes.
inline u32 RuntimeFormatElementSize(const RuntimeBufferFormat& format) {
    u32 bits = 0;
    for (u32 i = 0; i < 4; ++i) {
        const u32 width = (format.widths >> (i * 8)) & 0xff;
        if (width != 0) {
            bits = std::max(bits, ((format.offsets >> (i * 8)) & 0xff) + width);
        }
    }
    return bits / 8;
}

/// Whether a vertex attribute reads its format at run time. The vertex input then reads each
/// element as its raw byte, u16 or dwords, so the element must be aligned to that. The
/// specialisation, the translation and the vertex input all decide with this.
inline bool UsesRuntimeVertexFormat(bool runtime_vertex_formats, const AmdGpu::Buffer& buffer) {
    if (!runtime_vertex_formats) {
        return false;
    }
    const auto format = DescribeRuntimeBufferFormat(buffer);
    if (!format) {
        return false;
    }
    const u32 size = RuntimeFormatElementSize(*format);
    if (size == 0) {
        return false;
    }
    const u32 align = std::min(size, 4u);
    return buffer.base_address % align == 0 && buffer.GetStride() % align == 0;
}

/// With runtime vertex formats, how a vertex shader makes each component of an attribute
/// from the element's raw dwords, as an entry in the vertex stage table. The select is already
/// applied, so a shader converts only the components it reads.
struct RuntimeVertexFormat {
    enum class Mode : u32 {
        /// The bits as they are, sign-extended if signed.
        Raw = 0,
        /// 16-bit float.
        Half = 1,
        /// Converted to float and multiplied by the scale: normalised and scaled formats.
        Float = 2,
        /// Like Float, clamped to -1.
        Snorm = 3,
    };
    /// What a component made of no bits is.
    enum class Constant : u32 {
        Zero = 0,
        One = 1,
        IntegerOne = 2,
    };
    /// 16 bits per component, two components per word: dword, bit offset in it (or, with no
    /// width, a Constant), width, signed, mode.
    std::array<u32, 4> fields{};
    std::array<float, 4> scales{};

    static constexpr u32 DwordShift = 0;
    static constexpr u32 OffsetShift = 2;
    static constexpr u32 WidthShift = 7;
    static constexpr u32 SignedShift = 13;
    static constexpr u32 ModeShift = 14;

    void Set(u32 component, u32 dword, u32 offset, u32 width, bool is_signed, Mode mode,
             float scale) {
        const u32 field = dword << DwordShift | offset << OffsetShift | width << WidthShift |
                          u32(is_signed) << SignedShift | static_cast<u32>(mode) << ModeShift;
        fields[component / 2] |= field << (16 * (component % 2));
        scales[component] = scale;
    }
};
static_assert(sizeof(RuntimeVertexFormat) == 32);

/// Describes a vertex attribute's V# for a shader to convert at run time. Only for attributes
/// UsesRuntimeVertexFormat takes. Components the format lacks read like the vertex input fills
/// them: zero, and one for the fourth.
inline RuntimeVertexFormat DescribeRuntimeVertexFormat(const AmdGpu::Buffer& buffer) {
    using Kind = RuntimeBufferFormat::Kind;
    using Mode = RuntimeVertexFormat::Mode;
    using Constant = RuntimeVertexFormat::Constant;
    const auto format = *DescribeRuntimeBufferFormat(buffer);
    const auto kind = static_cast<Kind>(format.kind_swizzle & 0xf);
    const bool integer_one =
        AmdGpu::GetNumberClass(buffer.GetNumberFmt()) != AmdGpu::NumberClass::Float &&
        buffer.GetNumberConversion() == AmdGpu::NumberConversion::None;
    RuntimeVertexFormat out{};
    const auto constant = [&](u32 i, Constant c) {
        out.Set(i, 0, static_cast<u32>(c), 0, false, Mode::Raw, 0.f);
    };
    const auto select = buffer.DstSelect();
    for (u32 i = 0; i < 4; ++i) {
        const auto sel = select.array[i];
        if (sel == AmdGpu::CompSwizzle::Zero) {
            constant(i, Constant::Zero);
            continue;
        }
        if (sel == AmdGpu::CompSwizzle::One) {
            constant(i, Constant::One);
            continue;
        }
        const u32 c = static_cast<u32>(sel) - static_cast<u32>(AmdGpu::CompSwizzle::Red);
        const u32 width = (format.widths >> (c * 8)) & 0xff;
        if (c >= 4 || width == 0) {
            constant(i, c == 3 ? (integer_one ? Constant::IntegerOne : Constant::One)
                               : Constant::Zero);
            continue;
        }
        const u32 offset = (format.offsets >> (c * 8)) & 0xff;
        const double max = double((u64(1) << width) - 1);
        const double signed_max = width > 1 ? double((u64(1) << (width - 1)) - 1) : 1.0;
        switch (kind) {
        case Kind::Raw:
            out.Set(i, offset / 32, offset % 32, width, false, Mode::Raw, 1.f);
            break;
        case Kind::Sint:
            out.Set(i, offset / 32, offset % 32, width, true, Mode::Raw, 1.f);
            break;
        case Kind::Unorm:
            out.Set(i, offset / 32, offset % 32, width, false, Mode::Float, float(1.0 / max));
            break;
        case Kind::Snorm:
            out.Set(i, offset / 32, offset % 32, width, true, Mode::Snorm, float(1.0 / signed_max));
            break;
        case Kind::Half:
            out.Set(i, offset / 32, offset % 32, width, false, Mode::Half, 1.f);
            break;
        case Kind::Uscaled:
            out.Set(i, offset / 32, offset % 32, width, false, Mode::Float, 1.f);
            break;
        case Kind::Sscaled:
            out.Set(i, offset / 32, offset % 32, width, true, Mode::Float, 1.f);
            break;
        }
    }
    return out;
}

/// A storage image's swizzle for the stage table: the destination select of x, y, z and w in
/// 3 bits each, then its inverse for stores the same way from bit 12.
inline u32 EncodeImageSwizzle(const AmdGpu::CompMapping& swizzle) {
    u32 value = 0;
    const auto inverse = swizzle.Inverse();
    for (u32 i = 0; i < 4; ++i) {
        value |= (static_cast<u32>(swizzle.array[i]) & 7) << (i * 3);
        value |= (static_cast<u32>(inverse.array[i]) & 7) << (12 + i * 3);
    }
    return value;
}
static constexpr u32 ImageSwizzleInverseShift = 12;

/// Whether a buffer's formatted loads read its format at run time. The specialisation and the
/// translation both decide with this, so they always agree. Buffers with formatted stores
/// stay specialised.
inline bool UsesRuntimeBufferFormat(bool runtime_buffer_formats, const BufferResource& desc,
                                    const AmdGpu::Buffer& buffer) {
    return runtime_buffer_formats && desc.is_formatted && !desc.is_written &&
           DescribeRuntimeBufferFormat(buffer).has_value();
}

} // namespace Shader
