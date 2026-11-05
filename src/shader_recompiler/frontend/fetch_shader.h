// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/small_vector.h"
#include "common/types.h"
#include "shader_recompiler/info.h"

namespace Serialization {
struct Archive;
}

namespace Shader::Gcn {

struct VertexAttribute {
    enum InstanceIdType : u8 {
        None = 0,
        OverStepRate0 = 1,
        OverStepRate1 = 2,
        Plain = 3,
    };

    u8 dest_vgpr;     ///< Destination VGPR to load first component.
    u8 num_elements;  ///< Number of components to load
    u8 sgpr_base;     ///< SGPR that contains the pointer to the list of vertex V#
    u8 dword_offset;  ///< The dword offset of the V# that describes this attribute.
    u8 instance_data; ///< Indicates that the buffer will be accessed in instance rate
    u8 inst_offset;   ///< Instruction offset applied on the formatted buffer loads
    u8 data_format;   ///< Data format override when typed buffer loads are used
    u8 num_format;    ///< Number format override when typed buffer loads are used

    InstanceIdType GetStepRate() const {
        return static_cast<InstanceIdType>(instance_data);
    }

    constexpr AmdGpu::Buffer GetSharp(const Shader::Info& info) const noexcept {
        if (info.guess_state) {
            // No V# to read: assume 32-bit floats with as many components as the fetch loads,
            // the rest filled the way the formats fill them.
            auto buffer = AmdGpu::Buffer::Null();
            static constexpr std::array<AmdGpu::DataFormat, 5> Formats = {
                AmdGpu::DataFormat::FormatInvalid, AmdGpu::DataFormat::Format32,
                AmdGpu::DataFormat::Format32_32, AmdGpu::DataFormat::Format32_32_32,
                AmdGpu::DataFormat::Format32_32_32_32};
            buffer.data_format = u32(Formats[std::min<u32>(num_elements, 4)]);
            buffer.num_format = u32(AmdGpu::NumberFormat::Float);
            const auto select = [&](u32 i, AmdGpu::CompSwizzle fill) {
                return u32(i < num_elements ? AmdGpu::CompSwizzle(u32(AmdGpu::CompSwizzle::Red) + i)
                                            : fill);
            };
            buffer.dst_sel_x = select(0, AmdGpu::CompSwizzle::Zero);
            buffer.dst_sel_y = select(1, AmdGpu::CompSwizzle::Zero);
            buffer.dst_sel_z = select(2, AmdGpu::CompSwizzle::Zero);
            buffer.dst_sel_w = select(3, AmdGpu::CompSwizzle::One);
            return buffer;
        }
        auto buffer = info.ReadUdReg<AmdGpu::Buffer>(sgpr_base, dword_offset);
        buffer.base_address += inst_offset;
        if (data_format) {
            buffer.data_format = data_format;
            buffer.num_format = num_format;
        }
        return buffer;
    }

    bool operator==(const VertexAttribute& other) const = default;
};

struct FetchShaderData {
    u32 size = 0;
    SmallVector<VertexAttribute, 8> attributes;
    s8 vertex_offset_sgpr = -1;   ///< SGPR of vertex offset from VADDR
    s8 instance_offset_sgpr = -1; ///< SGPR of instance offset from VADDR

    bool operator==(const FetchShaderData& other) const {
        return attributes == other.attributes && vertex_offset_sgpr == other.vertex_offset_sgpr &&
               instance_offset_sgpr == other.instance_offset_sgpr;
    }

    bool Empty() const {
        return size == 0;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& buffer);
};

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base);

bool ParseFetchShader(const Shader::Info& info, FetchShaderData& out_fetch_data);

} // namespace Shader::Gcn
