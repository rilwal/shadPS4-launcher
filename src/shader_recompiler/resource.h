// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/ir/type.h"
#include "video_core/amdgpu/resource.h"

#include <array>
#include <bit>
#include <cstddef>
#include <span>

#include <boost/container/static_vector.hpp>

namespace Shader {

static constexpr u32 NUM_USER_DATA_REGS = 16;
static constexpr u32 NUM_IMAGES = 64;
static constexpr u32 NUM_BUFFERS = 56;
static constexpr u32 NUM_SAMPLERS = 16;
static constexpr u32 NUM_FMASKS = 8;

using SharpLocation = u16;

constexpr SharpLocation UNKNOWN_LOCATION = std::numeric_limits<u16>::max();

template <typename T>
struct SharpFetch {
    static constexpr std::size_t N = sizeof(T) / sizeof(u32);
    static_assert(N <= 8);

    enum class Summary : u8 {
        SingleLoad,
        MultiLoad,
        Invalid,
    };

    Summary summary{Summary::MultiLoad};
    std::array<u32, N> immediates;
    std::array<SharpLocation, N> offsets;
    u8 load_mask;

    bool operator==(const SharpFetch&) const = default;

    template <u32 num_dwords = N>
        requires(num_dwords <= N)
    constexpr bool Fetch(const u32* flatbuf, T* out) const {
        if (summary == Summary::SingleLoad) [[likely]] {
            std::memcpy(out, flatbuf + offsets[0], num_dwords * sizeof(u32));
            return true;
        } else if (summary == Summary::Invalid) [[unlikely]] {
            return false;
        }
        u8 mask = load_mask;
        std::array<u32, num_dwords> out_dw;
        for (u32 i = 0; i < num_dwords; i++) {
            out_dw[i] = (mask & 1) ? flatbuf[offsets[i]] : immediates[i];
            mask >>= 1;
        }
        std::memcpy(out, out_dw.data(), sizeof(out_dw));
        return true;
    }
};

enum class SharpFetchPostOp : u8 {
    None,
    // For buffers
    BitwiseOrDw1WithImm,
    OffsetByProgramBase,
    ClearStrideAndBitshiftDw2WithImm,
    // For images,
    ConvertCubeTo2DArray,
    // For samplers
    DisableAnisoIfSingleLod,
    ForceRepeatXyzClamp,
    ForceLastTexelXyClamp,
    ClearAnisoRatioAndThreshold,
};

enum class BufferType : u8 {
    Guest,
    Flatbuf,
    BdaPagetable,
    FaultBuffer,
    GdsBuffer,
    SharedMemory,
    ClipPlanes,
};

struct BufferResource {
    SharpFetch<AmdGpu::Buffer> sharp_fetch{};
    IR::Type used_types{};
    BufferType buffer_type{};
    bool is_written{};
    bool is_formatted{};
    SharpFetchPostOp post_op{};
    u32 post_op_dw1_mask{};
    u32 post_op_dw2_shift{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer;
        if (info.guess_state && buffer_type == BufferType::Guest) {
            // No V# to read: a bound buffer whose format the shader can read at run time, so
            // the specialisation is the one a draw with a describable format makes.
            buffer = AmdGpu::Buffer::Placeholder(1);
            buffer.data_format = u32(AmdGpu::DataFormat::Format32_32_32_32);
            buffer.num_format = u32(AmdGpu::NumberFormat::Float);
            buffer.dst_sel_x = u32(AmdGpu::CompSwizzle::Red);
            buffer.dst_sel_y = u32(AmdGpu::CompSwizzle::Green);
            buffer.dst_sel_z = u32(AmdGpu::CompSwizzle::Blue);
            buffer.dst_sel_w = u32(AmdGpu::CompSwizzle::Alpha);
            return buffer;
        }
        if (!sharp_fetch.Fetch(info.flattened_ud_buf.data(), &buffer)) {
            return AmdGpu::Buffer::Null();
        }
        if (post_op == SharpFetchPostOp::BitwiseOrDw1WithImm) {
            reinterpret_cast<u32*>(&buffer)[1] |= post_op_dw1_mask;
        } else if (post_op == SharpFetchPostOp::ClearStrideAndBitshiftDw2WithImm) {
            buffer.stride = 0;
            buffer.num_records <<= post_op_dw2_shift;
        } else if (post_op == SharpFetchPostOp::OffsetByProgramBase) {
            buffer.base_address += info.pgm_base;
        }
        if (!buffer.Valid()) {
            return AmdGpu::Buffer::Null();
        }
        return buffer;
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

enum class MipStorageFallbackMode : u16 {
    None,
    DynamicIndex,
    ConstantIndex,
};

/// A uniform condition the shader branches on before it reads a resource, as an expression over
/// the flattened user data. A table can hold anything in a slot the shader skips on this draw
/// (The Order: 1886 keeps 16 light probe slots and reads only the first `count`), so a resource
/// whose condition is false is bound as null instead of being built from whatever is there.
struct ResourcePredicate {
    enum class Op : u8 {
        Imm,  // imm
        Flat, // flattened_ud_buf[imm]
        IAdd,
        ISub,
        IMul,
        And,
        Or,
        Xor,
        Not,
        Shl,
        Shr,
        Sar,
        UMin,
        UMax,
        SMin,
        SMax,
        BfeU, // a >> b & mask(c)
        IEq,
        INe,
        ULt,
        ULe,
        UGt,
        UGe,
        SLt,
        SLe,
        SGt,
        SGe,
        LAnd,
        LOr,
        LXor,
        LNot,
        Select, // a ? b : c
        FAdd,
        FMul,
        FNeg,
        FAbs,
        FMin,
        FMax,
        FOrdEq,
        FOrdNe,
        FOrdLt,
        FOrdLe,
        FOrdGt,
        FOrdGe,
        FUnordEq,
        FUnordNe,
        FUnordLt,
        FUnordLe,
        FUnordGt,
        FUnordGe,
    };
    struct Node {
        Op op;
        u8 a, b, c; // operand nodes
        u32 imm;
    };
    static constexpr u32 MaxNodes = 255; // indices are u8, and roots are stored +1

    std::array<Node, MaxNodes> nodes;
    u8 num_nodes{};
    u16 max_flat{}; // highest flattened dword read, +1

    /// Whether the resource with predicate `root` (node index + 1, 0 for none) may be read.
    bool MayRead(u8 root, std::span<const u32> flatbuf) const noexcept {
        if (root == 0 || flatbuf.size() < max_flat) {
            return true;
        }
        return Eval(root - 1, flatbuf) != 0;
    }

private:
    u32 Eval(u8 index, std::span<const u32> flat) const noexcept {
        const Node& n = nodes[index];
        const auto u = [&](u8 i) { return Eval(i, flat); };
        const auto s = [&](u8 i) { return static_cast<s32>(Eval(i, flat)); };
        const auto f = [&](u8 i) { return std::bit_cast<float>(Eval(i, flat)); };
        const auto fu = [](float x) { return std::bit_cast<u32>(x); };
        const auto ord = [&](auto cmp) {
            const float x = f(n.a), y = f(n.b);
            return u32(x == x && y == y && cmp(x, y));
        };
        const auto unord = [&](auto cmp) {
            const float x = f(n.a), y = f(n.b);
            return u32(x != x || y != y || cmp(x, y));
        };
        switch (n.op) {
        case Op::Imm:
            return n.imm;
        case Op::Flat:
            return flat[n.imm];
        case Op::IAdd:
            return u(n.a) + u(n.b);
        case Op::ISub:
            return u(n.a) - u(n.b);
        case Op::IMul:
            return u(n.a) * u(n.b);
        case Op::And:
            return u(n.a) & u(n.b);
        case Op::Or:
            return u(n.a) | u(n.b);
        case Op::Xor:
            return u(n.a) ^ u(n.b);
        case Op::Not:
            return ~u(n.a);
        case Op::Shl:
            return u(n.a) << (u(n.b) & 31);
        case Op::Shr:
            return u(n.a) >> (u(n.b) & 31);
        case Op::Sar:
            return u32(s(n.a) >> (u(n.b) & 31));
        case Op::UMin:
            return std::min(u(n.a), u(n.b));
        case Op::UMax:
            return std::max(u(n.a), u(n.b));
        case Op::SMin:
            return u32(std::min(s(n.a), s(n.b)));
        case Op::SMax:
            return u32(std::max(s(n.a), s(n.b)));
        case Op::BfeU: {
            const u32 count = u(n.c) & 31;
            const u32 value = u(n.a) >> (u(n.b) & 31);
            return count == 0 ? 0 : value & (~0u >> (32 - count));
        }
        case Op::IEq:
            return u(n.a) == u(n.b);
        case Op::INe:
            return u(n.a) != u(n.b);
        case Op::ULt:
            return u(n.a) < u(n.b);
        case Op::ULe:
            return u(n.a) <= u(n.b);
        case Op::UGt:
            return u(n.a) > u(n.b);
        case Op::UGe:
            return u(n.a) >= u(n.b);
        case Op::SLt:
            return s(n.a) < s(n.b);
        case Op::SLe:
            return s(n.a) <= s(n.b);
        case Op::SGt:
            return s(n.a) > s(n.b);
        case Op::SGe:
            return s(n.a) >= s(n.b);
        case Op::LAnd:
            return u(n.a) && u(n.b);
        case Op::LOr:
            return u(n.a) || u(n.b);
        case Op::LXor:
            return (u(n.a) != 0) != (u(n.b) != 0);
        case Op::LNot:
            return !u(n.a);
        case Op::Select:
            return u(n.a) ? u(n.b) : u(n.c);
        case Op::FAdd:
            return fu(f(n.a) + f(n.b));
        case Op::FMul:
            return fu(f(n.a) * f(n.b));
        case Op::FNeg:
            return fu(-f(n.a));
        case Op::FAbs:
            return fu(std::abs(f(n.a)));
        case Op::FMin:
            return fu(std::min(f(n.a), f(n.b)));
        case Op::FMax:
            return fu(std::max(f(n.a), f(n.b)));
        case Op::FOrdEq:
            return ord([](float x, float y) { return x == y; });
        case Op::FOrdNe:
            return ord([](float x, float y) { return x != y; });
        case Op::FOrdLt:
            return ord([](float x, float y) { return x < y; });
        case Op::FOrdLe:
            return ord([](float x, float y) { return x <= y; });
        case Op::FOrdGt:
            return ord([](float x, float y) { return x > y; });
        case Op::FOrdGe:
            return ord([](float x, float y) { return x >= y; });
        case Op::FUnordEq:
            return unord([](float x, float y) { return x == y; });
        case Op::FUnordNe:
            return unord([](float x, float y) { return x != y; });
        case Op::FUnordLt:
            return unord([](float x, float y) { return x < y; });
        case Op::FUnordLe:
            return unord([](float x, float y) { return x <= y; });
        case Op::FUnordGt:
            return unord([](float x, float y) { return x > y; });
        case Op::FUnordGe:
            return unord([](float x, float y) { return x >= y; });
        }
        return 1;
    }
};

struct ImageResource {
    SharpFetch<AmdGpu::Image> sharp_fetch{};
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool is_r128{};
    u8 constant_mip_index{};
    MipStorageFallbackMode mip_fallback_mode{};
    SharpFetchPostOp post_op{};
    /// What a translation without the draw's T# takes the image to be, from how the shader
    /// addresses it.
    AmdGpu::ImageType guess_type{AmdGpu::ImageType::Color2D};
    /// Node of info.predicates (+1) that must hold for the shader to read the image; 0 if none.
    u8 predicate{};

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        AmdGpu::Image image{};
        if (info.guess_state) {
            image = AmdGpu::Image::Null(is_depth);
            // Non-null, so the specialisation takes it as bound like the draw's.
            image.base_address = 1;
            image.type = u64(post_op == SharpFetchPostOp::ConvertCubeTo2DArray
                                 ? AmdGpu::ImageType::Color2DArray
                                 : guess_type);
            return image;
        }
        if (!info.predicates.MayRead(predicate, info.flattened_ud_buf) ||
            !Fetch(info.flattened_ud_buf.data(), &image)) {
            return AmdGpu::Image::Null(is_depth);
        }
        if (post_op == SharpFetchPostOp::ConvertCubeTo2DArray) {
            image.type = u64(AmdGpu::ImageType::Color2DArray);
            image.depth = (image.depth + 1) * 6 - 1;
        }
        if (!image.Valid() || !image.IsPlausible()) {
            image = AmdGpu::Image::Null(is_depth);
        } else if (is_depth) {
            const auto data_fmt = image.GetDataFmt();
            if (data_fmt != AmdGpu::DataFormat::Format16 &&
                data_fmt != AmdGpu::DataFormat::Format32) {
                image = AmdGpu::Image::Null(true);
            }
        }
        return image;
    }

    constexpr bool Fetch(const u32* flatbuf, AmdGpu::Image* out) const {
        if (!is_r128) {
            // Fetch full 8 byte T#
            return sharp_fetch.Fetch(flatbuf, out);
        }
        // Fetch r128 T# and fix pitch
        if (!sharp_fetch.Fetch<4>(flatbuf, out)) {
            return false;
        }
        out->pitch = out->width;
        return true;
    }

    u32 NumBindings(const auto& info) const {
        const AmdGpu::Image tsharp = GetSharp(info);
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex)
                   ? (tsharp.last_level - tsharp.base_level + 1)
                   : 1;
    }
};
using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

struct SamplerResource {
    SharpFetch<AmdGpu::Sampler> sharp_fetch{};
    SharpFetchPostOp post_op{};
    SharpLocation post_op_tsharp_dw3_off{};
    bool is_depth{};

    constexpr AmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        AmdGpu::Sampler sampler{};
        sharp_fetch.Fetch(info.flattened_ud_buf.data(), &sampler);
        if (post_op == SharpFetchPostOp::DisableAnisoIfSingleLod) {
            const u32 tsharp_dw3 = info.flattened_ud_buf[post_op_tsharp_dw3_off];
            if (((tsharp_dw3 >> 12) & 0xff) == 0) {
                sampler.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        } else if (post_op == SharpFetchPostOp::ForceRepeatXyzClamp) {
            sampler.clamp_x.Assign(AmdGpu::ClampMode::Wrap);
            sampler.clamp_y.Assign(AmdGpu::ClampMode::Wrap);
            sampler.clamp_z.Assign(AmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ForceLastTexelXyClamp) {
            sampler.clamp_x.Assign(AmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_y.Assign(AmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_z.Assign(AmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ClearAnisoRatioAndThreshold) {
            sampler.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            sampler.aniso_threshold.Assign(0);
        }
        return sampler;
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

struct FMaskResource {
    SharpLocation sharp_idx;

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        return info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
    }
};
using FMaskResourceList = boost::container::static_vector<FMaskResource, NUM_FMASKS>;

struct PushData {
    static constexpr u32 XOffsetIndex = 0;
    static constexpr u32 YOffsetIndex = 1;
    static constexpr u32 XScaleIndex = 2;
    static constexpr u32 YScaleIndex = 3;
    static constexpr u32 BufOffsetIndex = 4;

    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    std::array<u8, NUM_BUFFERS> buf_offsets;

    void AddOffset(u32 binding, u32 offset) {
        ASSERT(offset < 256 && binding < buf_offsets.size());
        buf_offsets[binding] = offset;
    }
};
static_assert(sizeof(PushData) <= 128,
              "PushData size is greater than minimum size guaranteed by Vulkan spec");

/// With VK_EXT_descriptor_heap, each stage of a draw has a table in memory, at an address in
/// push data: its buffer offsets, then one heap index per descriptor it reads.
struct StageTable {
    /// The table is also a uniform buffer the shader declares at this set and binding.
    static constexpr u32 DescriptorSet = 1;
    static constexpr u32 Binding = 0;
    /// Buffer offsets as bytes, read as four uvec4s.
    static constexpr u32 BufOffsetsSize = 64;
    /// Buffer strides as u16s, read as eight uvec4s after the offsets: shaders read strides
    /// at run time, so one shader serves buffers of any stride.
    static constexpr u32 StridesOffset = BufOffsetsSize;
    static constexpr u32 StridesSize = 128;
    static constexpr u32 StridesMember = 4;
    /// With runtime buffer formats, a RuntimeBufferFormat (16 bytes) per buffer, read as one
    /// uvec4 each.
    static constexpr u32 FormatsOffset = StridesOffset + StridesSize;
    static constexpr u32 FormatsSize = NUM_BUFFERS * 16;
    static constexpr u32 FormatsMember = StridesMember + StridesSize / 16;
    /// With runtime storage swizzles, each storage image's swizzle as a u32 (see
    /// EncodeImageSwizzle), indexed by image resource and read as uvec4s.
    static constexpr u32 ImageSwizzlesOffset = FormatsOffset + FormatsSize;
    static constexpr u32 ImageSwizzlesSize = NUM_IMAGES * sizeof(u32);
    static constexpr u32 ImageSwizzlesMember = FormatsMember + 1;
    /// With runtime target swizzles, each render target's swizzle as a u32 (the forward half
    /// of EncodeImageSwizzle), read as uvec4s. Fragment stages only.
    static constexpr u32 TargetSwizzlesOffset = ImageSwizzlesOffset + ImageSwizzlesSize;
    static constexpr u32 TargetSwizzlesSize = 8 * sizeof(u32);
    static constexpr u32 TargetSwizzlesMember = ImageSwizzlesMember + 1;
    /// With runtime vertex formats, a RuntimeVertexFormat (32 bytes) per vertex attribute,
    /// read as two uvec4s each. Vertex stages only.
    static constexpr u32 VertexFormatsOffset = TargetSwizzlesOffset + TargetSwizzlesSize;
    static constexpr u32 NumVertexFormats = 32;
    static constexpr u32 VertexFormatsSize = NumVertexFormats * 32;
    static constexpr u32 VertexFormatsMember = TargetSwizzlesMember + 1;
    static constexpr u32 IndicesOffset = VertexFormatsOffset + VertexFormatsSize;
    /// Room for every buffer, image binding and sampler a stage can have.
    static constexpr u32 MaxIndices = NUM_BUFFERS + NUM_IMAGES * 16 + NUM_SAMPLERS;
    static constexpr u32 MaxSize = IndicesOffset + MaxIndices * sizeof(u32);
};
static_assert(NUM_BUFFERS <= StageTable::BufOffsetsSize);
static_assert(NUM_BUFFERS <= StageTable::StridesSize / sizeof(u16));

/// Push data with VK_EXT_descriptor_heap: the push constants above, then each logical
/// stage's table address.
struct HeapPushData {
    static constexpr u32 NumStages = 6;

    PushData data;
    alignas(8) std::array<u64, NumStages> stage_tables;

    static constexpr u32 StageTableOffset(u32 stage) {
        return static_cast<u32>(offsetof(HeapPushData, stage_tables) + stage * sizeof(u64));
    }
};
static_assert(sizeof(HeapPushData) <= 256,
              "HeapPushData is larger than the push data VK_EXT_descriptor_heap guarantees");

} // namespace Shader
