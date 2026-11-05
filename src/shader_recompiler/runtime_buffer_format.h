// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

/// Whether a buffer's formatted loads read its format at run time. The specialisation and the
/// translation both decide with this, so they always agree. Buffers with formatted stores
/// stay specialised.
inline bool UsesRuntimeBufferFormat(bool runtime_buffer_formats, const BufferResource& desc,
                                    const AmdGpu::Buffer& buffer) {
    return runtime_buffer_formats && desc.is_formatted && !desc.is_written &&
           DescribeRuntimeBufferFormat(buffer).has_value();
}

} // namespace Shader
