// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/specialization.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/resource.h"

/// What a shader is translated for, worked out from the guest's registers, from a recorded
/// draw state or guessed from the shader's own PSSL struct. Needs no device, so the shader
/// tool shares it with the pipeline cache.
namespace Vulkan {

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

/// What the runtime info depends on besides the guest: the recompiler profile and the device
/// features that shape it.
struct ShaderHost {
    const Shader::Profile* profile;
    bool depth_clip_control;
    bool vertex_input_dynamic_state;
};

/// A host saved next to a game's shader list, so the shader tool translates as the emulator
/// last did. Raw, so only builds with the same Profile layout read it.
struct SavedShaderHost {
    Shader::Profile profile;
    bool depth_clip_control;
    bool vertex_input_dynamic_state;
    /// The emulator setting, which changes how shaders read memory.
    bool direct_memory_access;

    ShaderHost Host() const {
        return {&profile, depth_clip_control, vertex_input_dynamic_state};
    }
};
bool WriteShaderHost(const std::filesystem::path& path, const ShaderHost& host,
                     bool direct_memory_access);
std::optional<SavedShaderHost> ReadShaderHost(const std::filesystem::path& path);

using ColorBuffers = std::array<Shader::PsColorBuffer, AmdGpu::NUM_COLOR_BUFFERS>;

/// Builds a stage's runtime info from the registers, so a draw recorded as its registers can
/// be built again later. The compute registers are needed for compute, the hull shader's info
/// for the local stage.
void FillRuntimeInfo(const ShaderHost& host, Shader::RuntimeInfo& info, Shader::HwStage stage,
                     Shader::SwStage l_stage, const AmdGpu::Regs& regs,
                     const AmdGpu::ComputeProgram* cs_regs, const ColorBuffers& color_buffers,
                     const DrawIndirectParams& indirect, const Shader::Info* hull_info);
/// The render targets as the shader recompiler sees them, before draws mask them.
void FillColorBuffers(const AmdGpu::Regs& regs, ColorBuffers& color_buffers);
/// A render target as the fragment shader is specialized on it.
Shader::PsColorBuffer ShaderColorBuffer(const Shader::Profile& profile,
                                        const Shader::PsColorBuffer& target);
Shader::PsColorBuffer UnboundTarget(AmdGpu::ShaderExportFormat export_format);
u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl);

std::vector<std::string> SpecDifferences(const Shader::StageSpecialization& a,
                                         const Shader::StageSpecialization& b);

/// The stages whose register state a shader's struct gives.
bool PrecompilesStage(Shader::HwStage stage);
bool GuessTranslated(const Shader::Info& info, Shader::HwStage hw_stage);
/// The register state a shader's PSSL struct gives, for translating it before any draw uses
/// it: the stage registers as written there, with the rest of the draw guessed. A pixel
/// shader's inputs are its semantics less `semantic_base` (each shader's lowest when it's -1).
std::optional<Shader::RuntimeInfo> GuessRuntimeInfo(const ShaderHost& host,
                                                    Shader::HwStage hw_stage,
                                                    Shader::SwStage sw_stage,
                                                    std::span<const u8> header, int semantic_base);
std::optional<Shader::Gcn::FetchShaderData> GuessFetchShader(std::span<const u8> header);
std::optional<std::vector<u16>> PsInputSemantics(std::span<const u8> header);
std::vector<Shader::Info::GuessedImage> ReflectedImages(Shader::HwStage stage,
                                                        std::span<const u8> header,
                                                        std::span<const u8> reflection);

/// Recorded draw states, in terms of the guest's own hardware rather than this build's structs,
/// so they keep working across emulator changes and can be shared: the registers (as register
/// numbers and values), the draw's indirect parameters, the user data, the flattened user data
/// with what identifies its layout, the fetch shader's attributes and the vertex buffer V#s.
/// They hold no shader code. A blob starts with the format and a key for what it records, which
/// the scanner dedupes on (ShaderFiles::StateKeyBytes), then the stage.
constexpr u32 DrawStateFormat = 3;

/// The register ranges recorded: config and shader registers, context registers, and user
/// config registers. Only nonzero registers are kept.
constexpr std::array<std::pair<u32, u32>, 3> DrawStateRegRanges{{
    {AmdGpu::Regs::ConfigRegWordOffset, AmdGpu::Regs::ShRegWordOffset + 0x400},
    {AmdGpu::Regs::ContextRegWordOffset, AmdGpu::Regs::ContextRegWordOffset + 0x400},
    {AmdGpu::Regs::UconfigRegWordOffset, AmdGpu::Regs::NumRegs},
}};

/// The compute registers kept, from the dispatch initiator up to the user data.
constexpr u32 ComputeRegDwords = offsetof(AmdGpu::ComputeProgram, user_data) / sizeof(u32);

template <typename T>
void PutState(std::vector<u8>& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
bool GetState(std::span<const u8>& in, T& value) {
    if (in.size() < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, in.data(), sizeof(T));
    in = in.subspan(sizeof(T));
    return true;
}

/// A recorded draw state, rebuilt.
struct DrawState {
    Shader::HwStage hw_stage;
    Shader::SwStage sw_stage;
    Shader::RuntimeInfo runtime_info;
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data;
    u64 flat_layout;
    std::vector<u32> flat_buf;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::vector<std::pair<u32, AmdGpu::Buffer>> vertex_buffers;
};
std::optional<DrawState> ParseDrawState(const ShaderHost& host, std::span<const u8> blob);

} // namespace Vulkan
