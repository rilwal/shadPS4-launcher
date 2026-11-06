// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <fstream>
#include <memory>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/shader_guess.h"

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

namespace {
constexpr u32 HostMagic = 0x54534853; // "SHST"
}

bool WriteShaderHost(const std::filesystem::path& path, const ShaderHost& host,
                     bool direct_memory_access) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    const u32 size = sizeof(Shader::Profile);
    const u8 flags[3] = {host.depth_clip_control, host.vertex_input_dynamic_state,
                         direct_memory_access};
    file.write(reinterpret_cast<const char*>(&HostMagic), sizeof(HostMagic));
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(host.profile), sizeof(Shader::Profile));
    file.write(reinterpret_cast<const char*>(flags), sizeof(flags));
    return bool(file);
}

std::optional<SavedShaderHost> ReadShaderHost(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    u32 magic{};
    u32 size{};
    u8 flags[3]{};
    SavedShaderHost saved{};
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!file || magic != HostMagic || size != sizeof(Shader::Profile)) {
        return std::nullopt;
    }
    file.read(reinterpret_cast<char*>(&saved.profile), sizeof(saved.profile));
    file.read(reinterpret_cast<char*>(flags), sizeof(flags));
    if (!file) {
        return std::nullopt;
    }
    saved.depth_clip_control = flags[0] != 0;
    saved.vertex_input_dynamic_state = flags[1] != 0;
    saved.direct_memory_access = flags[2] != 0;
    return saved;
}

/// Names the parts of two specializations of one shader that differ, so the log shows what each
/// new variant was compiled for. `a` is the new one: like the lookup, only resources it binds
/// are compared.
std::vector<std::string> SpecDifferences(const Shader::StageSpecialization& a,
                                         const Shader::StageSpecialization& b) {
    std::vector<std::string> out;
    const auto check = [&](bool same, const char* name, auto... index) {
        if (!same) {
            out.push_back(fmt::format(fmt::runtime(name), index...));
        }
    };
    const auto& ra = a.runtime_info;
    const auto& rb = b.runtime_info;
    // Each field with both values, since a value pair says where a guess goes wrong.
    const auto prop = [&](auto pa, auto pb, const char* name) {
        if (pa != pb) {
            out.push_back(fmt::format("props.{} {}/{}", name, u32(pa), u32(pb)));
        }
    };
    const auto& pa = ra.props;
    const auto& pb = rb.props;
    prop(pa.fp_denorm_mode32, pb.fp_denorm_mode32, "fp_denorm_mode32");
    prop(pa.fp_denorm_mode16_64, pb.fp_denorm_mode16_64, "fp_denorm_mode16_64");
    prop(pa.fp_round_mode32, pb.fp_round_mode32, "fp_round_mode32");
    prop(pa.fp_round_mode16_64, pb.fp_round_mode16_64, "fp_round_mode16_64");
    prop(pa.num_allocated_vgprs, pb.num_allocated_vgprs, "num_allocated_vgprs");
    prop(pa.num_user_data, pb.num_user_data, "num_user_data");
    prop(pa.num_input_vgprs, pb.num_input_vgprs, "num_input_vgprs");
    prop(pa.dx10_clamp, pb.dx10_clamp, "dx10_clamp");
    if (ra.sw_stage == SwStage::Vertex) {
        const auto& va = ra.sw.vs;
        const auto& vb = rb.sw.vs;
        check(va.step_rate_0 == vb.step_rate_0 && va.step_rate_1 == vb.step_rate_1,
              "vs.step_rates");
        check(va.vertex_sgpr_offset == vb.vertex_sgpr_offset &&
                  va.instance_sgpr_offset == vb.instance_sgpr_offset,
              "vs.sgpr_offsets");
        check(va.is_indirect == vb.is_indirect, "vs.is_indirect");
        check(va.tess_emulated_primitive == vb.tess_emulated_primitive, "vs.tess_emulated");
    } else if (!ra.HasSameSwInfo(rb.sw)) {
        out.emplace_back("sw_info");
    }
    switch (ra.hw_stage) {
    case HwStage::Fragment: {
        const auto& fa = ra.hw.fs;
        const auto& fb = rb.hw.fs;
        for (u32 i = 0; i < Shader::MaxColorBuffers; ++i) {
            const auto& ca = fa.color_buffers[i];
            const auto& cb = fb.color_buffers[i];
            check(ca.data_format == cb.data_format && ca.num_format == cb.num_format,
                  "fs.cb{}.format", i);
            check(ca.num_conversion == cb.num_conversion, "fs.cb{}.num_conversion", i);
            check(ca.export_format == cb.export_format, "fs.cb{}.export_format", i);
            check(ca.blend_self_scale == cb.blend_self_scale, "fs.cb{}.blend_self_scale", i);
            check(ca.swizzle == cb.swizzle, "fs.cb{}.swizzle", i);
        }
        check(fa.en_flags == fb.en_flags, "fs.en_flags");
        check(fa.addr_flags == fb.addr_flags, "fs.addr_flags");
        check(fa.num_inputs == fb.num_inputs &&
                  std::ranges::equal(fa.inputs.begin(), fa.inputs.begin() + fa.num_inputs,
                                     fb.inputs.begin(), fb.inputs.begin() + fb.num_inputs),
              "fs.inputs");
        check(fa.z_export_format == fb.z_export_format, "fs.z_export_format");
        check(fa.num_samples == fb.num_samples, "fs.num_samples");
        check(fa.mrtz_mask == fb.mrtz_mask, "fs.mrtz_mask");
        check(fa.front_face_all_bits == fb.front_face_all_bits, "fs.front_face_all_bits");
        check(fa.dual_source_blending == fb.dual_source_blending, "fs.dual_source_blending");
        check(fa.clip_distance_emulation == fb.clip_distance_emulation,
              "fs.clip_distance_emulation");
        check(fa.depth_before_shader == fb.depth_before_shader, "fs.depth_before_shader");
        break;
    }
    case HwStage::Vertex: {
        const auto& va = ra.hw.vs;
        const auto& vb = rb.hw.vs;
        check(va.num_outputs == vb.num_outputs && va.outputs == vb.outputs, "hwvs.outputs");
        check(va.emulate_depth_negative_one_to_one == vb.emulate_depth_negative_one_to_one,
              "hwvs.depth_negative_one_to_one");
        check(va.clip_disable == vb.clip_disable, "hwvs.clip_disable");
        check(va.user_clip_plane_mask == vb.user_clip_plane_mask, "hwvs.user_clip_planes");
        break;
    }
    case HwStage::Compute: {
        const auto& ca = ra.hw.cs;
        const auto& cb = rb.hw.cs;
        check(ca.workgroup_size == cb.workgroup_size, "cs.workgroup_size");
        check(ca.tgid_enable == cb.tgid_enable, "cs.tgid_enable");
        break;
    }
    default:
        check(ra.HasSameHwInfo(rb.hw), "hw_info");
        break;
    }
    check(a.fetch_shader_data == b.fetch_shader_data, "fetch_shader");
    for (u32 i = 0; i < std::min(a.vs_attribs.size(), b.vs_attribs.size()); ++i) {
        const auto& x = a.vs_attribs[i];
        const auto& y = b.vs_attribs[i];
        check(x.divisor == y.divisor, "attr{}.divisor", i);
        check(x.num_class == y.num_class, "attr{}.num_class", i);
        check(x.dst_select == y.dst_select, "attr{}.dst_select", i);
        check(x.runtime_format == y.runtime_format, "attr{}.runtime_format", i);
    }
    check(a.vs_attribs.size() == b.vs_attribs.size(), "attr.count");
    check(a.fmasks == b.fmasks, "fmasks");
    const bool has_resources = !a.buffers.empty() || !a.images.empty() || !a.samplers.empty();
    if (has_resources) {
        check(a.start == b.start, "binding_start");
    }
    u32 binding{};
    for (u32 i = 0; i < a.buffers.size() && i < b.buffers.size(); ++i) {
        if (!a.bitset[binding++]) {
            continue;
        }
        const auto& x = a.buffers[i];
        const auto& y = b.buffers[i];
        check(x.stride == y.stride, "buf{}.stride", i);
        check(x.is_formatted == y.is_formatted, "buf{}.is_formatted", i);
        check(x.runtime_format == y.runtime_format, "buf{}.runtime_format", i);
        if (x.is_formatted && y.is_formatted && !x.runtime_format && !y.runtime_format) {
            check(x.data_format == y.data_format && x.num_format == y.num_format, "buf{}.format",
                  i);
            check(x.dst_select == y.dst_select, "buf{}.dst_select", i);
            check(x.num_conversion == y.num_conversion, "buf{}.num_conversion", i);
        }
        check(x.swizzle_enable == y.swizzle_enable, "buf{}.swizzle_enable", i);
        if (x.swizzle_enable && y.swizzle_enable) {
            check(x.index_stride == y.index_stride && x.element_size == y.element_size,
                  "buf{}.swizzle", i);
        }
    }
    binding = static_cast<u32>(a.buffers.size());
    for (u32 i = 0; i < a.images.size() && i < b.images.size(); ++i) {
        if (!a.bitset[binding++]) {
            continue;
        }
        const auto& x = a.images[i];
        const auto& y = b.images[i];
        if (x.type != y.type) {
            out.push_back(fmt::format("img{}.type {}/{}{}", i, magic_enum::enum_name(x.type),
                                      magic_enum::enum_name(y.type),
                                      x.is_storage ? " storage" : ""));
        }
        check(x.is_integer == y.is_integer, "img{}.is_integer", i);
        check(x.is_storage == y.is_storage, "img{}.is_storage", i);
        check(x.is_srgb == y.is_srgb, "img{}.is_srgb", i);
        check(x.dst_select == y.dst_select, "img{}.dst_select", i);
        check(x.num_conversion == y.num_conversion, "img{}.num_conversion", i);
        check(x.num_bindings == y.num_bindings, "img{}.num_bindings", i);
    }
    for (u32 i = 0; i < a.samplers.size() && i < b.samplers.size(); ++i) {
        check(a.samplers[i].force_unnormalized == b.samplers[i].force_unnormalized,
              "smp{}.force_unnormalized", i);
        check(a.samplers[i].force_degamma == b.samplers[i].force_degamma, "smp{}.force_degamma", i);
    }
    return out;
}

u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

/// The render target a pixel shader is taken to export to where nothing says what's bound:
/// 8-bit unorm in RGBA order. Used for guesses, and with run-time target swizzles for exported
/// locations the draw leaves unbound, so that the two agree.
Shader::PsColorBuffer UnboundTarget(AmdGpu::ShaderExportFormat export_format) {
    Shader::PsColorBuffer target{
        .data_format = AmdGpu::DataFormat::Format8_8_8_8,
        .num_format = AmdGpu::NumberFormat::Unorm,
        .num_conversion = AmdGpu::NumberConversion::None,
        .export_format = export_format,
        .blend_self_scale = 0,
    };
    target.swizzle.array = {AmdGpu::CompSwizzle::Red, AmdGpu::CompSwizzle::Green,
                            AmdGpu::CompSwizzle::Blue, AmdGpu::CompSwizzle::Alpha};
    return target;
}

void FillRuntimeInfo(const ShaderHost& host, Shader::RuntimeInfo& info, HwStage stage,
                     SwStage l_stage, const AmdGpu::Regs& regs,
                     const AmdGpu::ComputeProgram* cs_regs, const ColorBuffers& color_buffers,
                     const DrawIndirectParams& indirect, const Shader::Info* hull_info) {
    const auto BuildCommon = [&](const auto& program) {
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.dx10_clamp = program.settings.dx10_clamp;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        ASSERT(hull_info);
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            info.hw.gs.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !host.depth_clip_control &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.depth_before_shader = regs.depth_shader_control.depth_before_shader;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u16(ps_inputs[i].input_offset),
                .is_default = u16(ps_inputs[i].use_default),
                .is_flat = u16(ps_inputs[i].flat_shade),
                .default_value = u16(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            auto target = color_buffers[i];
            if (host.profile->runtime_target_swizzles && !info.hw.fs.dual_source_blending) {
                // The shader exports what its export format says whatever the draw binds;
                // exports to a location without an attachment are dropped.
                target.export_format = regs.color_export_format.GetFormat(i);
                // An exported location the draw leaves unbound takes the same stand-in as a
                // guess: its exports are dropped either way.
                if (target.export_format == AmdGpu::ShaderExportFormat::Zero) {
                    // Not exported to, so what's bound there doesn't reach the code.
                    target = {};
                } else if (target.data_format == AmdGpu::DataFormat::FormatInvalid) {
                    target = UnboundTarget(target.export_format);
                }
            }
            info.hw.fs.color_buffers[i] = host.profile->fewer_shader_variants
                                              ? ShaderColorBuffer(*host.profile, target)
                                              : target;
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            host.profile->needs_clip_distance_emulation;
        break;
    }
    case HwStage::Compute: {
        ASSERT(cs_regs);
        const auto& cs_pgm = *cs_regs;
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.props.dx10_clamp = cs_pgm.settings.dx10_clamp;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        // A step rate only reaches the code through the instance ID VGPR it divides, so a
        // shader that doesn't take that VGPR is the same code at any rate. Pipelines only take
        // the rates from here without dynamic vertex input.
        if (host.profile->fewer_shader_variants && host.vertex_input_dynamic_state) {
            if (info.props.num_input_vgprs < 1) {
                info.sw.vs.step_rate_0 = 0;
            }
            if (info.props.num_input_vgprs < 2) {
                info.sw.vs.step_rate_1 = 0;
            }
            // The count also takes the fetch shader's instance inputs, set by the fetch shader
            // generator at bind time and missing from the shader struct, and those reach the
            // shader as vertex inputs. With every instance VGPR set the code is the same: a
            // register the hardware leaves unset is one the shader writes before reading.
            if (stage == HwStage::Vertex) {
                info.props.num_input_vgprs = 3;
            }
        }
        info.sw.vs.vertex_sgpr_offset = indirect.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = indirect.instance_sgpr_offset;
        info.sw.vs.is_indirect = indirect.is_indirect;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
}

void FillColorBuffers(const AmdGpu::Regs& regs, ColorBuffers& color_buffers) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable) {
        return;
    }
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }
}

/// The parts of a render target that change a fragment shader's code. The output type follows
/// the number class, not the exact format, and a swizzle's unused components map to
/// themselves, since the target drops them anyway. Keeping only these lets one shader serve
/// targets of different formats.
Shader::PsColorBuffer ShaderColorBuffer(const Shader::Profile& profile,
                                        const Shader::PsColorBuffer& target) {
    Shader::PsColorBuffer buffer = target;
    if (!profile.needs_unorm_fixup) {
        // The fix-up is the only use of the exact format.
        buffer.data_format = AmdGpu::DataFormat::FormatInvalid;
        switch (AmdGpu::GetNumberClass(target.num_format)) {
        case AmdGpu::NumberClass::Float:
            buffer.num_format = AmdGpu::NumberFormat::Float;
            break;
        case AmdGpu::NumberClass::Sint:
            buffer.num_format = AmdGpu::NumberFormat::Sint;
            break;
        case AmdGpu::NumberClass::Uint:
            buffer.num_format = AmdGpu::NumberFormat::Uint;
            break;
        }
    }
    for (u32 i = 0; i < 4; ++i) {
        // With runtime target swizzles the shader reads the order from the stage table.
        const u32 component = profile.runtime_target_swizzles ? i : target.swizzle.Map(i);
        buffer.swizzle.array[i] =
            static_cast<AmdGpu::CompSwizzle>(u32(AmdGpu::CompSwizzle::Red) + component);
    }
    return buffer;
}

/// Whether a translation from a guessed stage came out as something that stage can be. A
/// guess of the wrong stage mustn't reach the backend's checks, which end the emulator: shared
/// memory on a graphics stage means the shader is really a hull, local or export shader.
bool GuessTranslated(const Shader::Info& info, HwStage hw_stage) {
    return !info.translation_failed &&
           (hw_stage == HwStage::Compute || static_cast<u32>(info.shared_types) == 0);
}

/// The stages whose register state a shader's struct gives.
bool PrecompilesStage(HwStage stage) {
    return stage == HwStage::Fragment || stage == HwStage::Compute || stage == HwStage::Vertex;
}

/// Image types from a shader's PSSL reflection, by where the shader reads each read-only
/// texture's T#: the struct's usage slots say which user data registers hold a texture
/// (type 0x00, by api slot) or point at a resource table (0x13, 0x1b; api slot k at dword 8k).
std::vector<Shader::Info::GuessedImage> ReflectedImages(HwStage stage, std::span<const u8> header,
                                                        std::span<const u8> reflection) {
    constexpr size_t RecordSize = 24;
    // The usage slots follow the stage registers, with an input count word between for the
    // vertex and pixel stages.
    const size_t slots = stage == HwStage::Fragment ? 8 + 12 * 4 + 4
                         : stage == HwStage::Vertex ? 8 + 7 * 4 + 4
                                                    : 8 + 7 * 4;
    if (reflection.empty() || header.size() < 4) {
        return {};
    }
    u32 common;
    std::memcpy(&common, header.data(), sizeof(common));
    const u32 num_usage_slots = common >> 24;
    if (slots + num_usage_slots * 4 > header.size()) {
        return {};
    }
    std::vector<Shader::Info::GuessedImage> images;
    for (size_t r = 0; r + RecordSize <= reflection.size(); r += RecordSize) {
        u32 api_slot;
        u32 type_word;
        std::memcpy(&api_slot, reflection.data() + r, sizeof(api_slot));
        std::memcpy(&type_word, reflection.data() + r + 8, sizeof(type_word));
        const u8 dimension = type_word & 0xff;
        const u8 kind = (type_word >> 8) & 0xff;
        const u8 access = (type_word >> 16) & 0xff;
        if (kind != 0x05 || access != 0x03) {
            continue;
        }
        AmdGpu::ImageType type;
        switch (dimension) {
        case 0x01:
            type = AmdGpu::ImageType::Color1D;
            break;
        case 0x02:
            type = AmdGpu::ImageType::Color2D;
            break;
        case 0x03:
            type = AmdGpu::ImageType::Color3D;
            break;
        case 0x04:
        case 0x07:
            type = AmdGpu::ImageType::Cube;
            break;
        case 0x06:
            type = AmdGpu::ImageType::Color2DArray;
            break;
        case 0x08:
            type = AmdGpu::ImageType::Color2DMsaa;
            break;
        default:
            continue;
        }
        for (u32 i = 0; i < num_usage_slots; ++i) {
            const u8* usage = header.data() + slots + i * 4;
            if (usage[0] == 0x00 && usage[1] == api_slot) {
                images.push_back({.sgpr = usage[2], .table = false, .dword = 0, .type = type});
            } else if (usage[0] == 0x13 || usage[0] == 0x1b) {
                images.push_back(
                    {.sgpr = usage[2], .table = true, .dword = api_slot * 8, .type = type});
            }
        }
    }
    return images;
}

std::optional<DrawState> ParseDrawState(const ShaderHost& host, std::span<const u8> in) {
    u32 format{};
    u64 key{};
    u8 hw{};
    u8 sw{};
    if (!GetState(in, format) || format != DrawStateFormat || !GetState(in, key) ||
        !GetState(in, hw) || !GetState(in, sw)) {
        return std::nullopt;
    }
    DrawState state{};
    state.hw_stage = static_cast<HwStage>(hw);
    state.sw_stage = static_cast<SwStage>(sw);
    // Only the stages compiled early: the others need more than registers (the hull shader's
    // constants, the geometry stage's copy shader).
    if (state.hw_stage != HwStage::Vertex && state.hw_stage != HwStage::Fragment &&
        state.hw_stage != HwStage::Compute) {
        return std::nullopt;
    }
    thread_local auto regs = std::make_unique<AmdGpu::Regs>();
    std::ranges::fill(regs->reg_array, 0u);
    u32 num_regs{};
    if (!GetState(in, num_regs) || num_regs > AmdGpu::Regs::NumRegs) {
        return std::nullopt;
    }
    for (u32 i = 0; i < num_regs; ++i) {
        u16 index{};
        u32 value{};
        if (!GetState(in, index) || !GetState(in, value) || index >= AmdGpu::Regs::NumRegs) {
            return std::nullopt;
        }
        regs->reg_array[index] = value;
    }
    AmdGpu::ComputeProgram cs_regs{};
    u32 num_cs_dwords{};
    if (!GetState(in, num_cs_dwords) || num_cs_dwords > ComputeRegDwords ||
        in.size() < num_cs_dwords * sizeof(u32)) {
        return std::nullopt;
    }
    std::memcpy(&cs_regs, in.data(), num_cs_dwords * sizeof(u32));
    in = in.subspan(num_cs_dwords * sizeof(u32));
    DrawIndirectParams indirect{};
    u8 is_indirect{};
    if (!GetState(in, indirect.vertex_sgpr_offset) ||
        !GetState(in, indirect.instance_sgpr_offset) || !GetState(in, is_indirect) ||
        !GetState(in, state.user_data) || !GetState(in, state.flat_layout)) {
        return std::nullopt;
    }
    indirect.is_indirect = is_indirect != 0;
    u32 flat_size{};
    if (!GetState(in, flat_size) || in.size() < flat_size * sizeof(u32)) {
        return std::nullopt;
    }
    state.flat_buf.resize(flat_size);
    std::memcpy(state.flat_buf.data(), in.data(), flat_size * sizeof(u32));
    in = in.subspan(flat_size * sizeof(u32));
    u32 fetch_size{};
    if (!GetState(in, fetch_size)) {
        return std::nullopt;
    }
    if (fetch_size != 0) {
        auto& fetch = state.fetch.emplace();
        fetch.size = fetch_size;
        u32 num_attributes{};
        if (!GetState(in, fetch.vertex_offset_sgpr) || !GetState(in, fetch.instance_offset_sgpr) ||
            !GetState(in, num_attributes) || num_attributes > Shader::IR::NumParams) {
            return std::nullopt;
        }
        for (u32 i = 0; i < num_attributes; ++i) {
            std::array<u8, 8> fields{};
            if (!GetState(in, fields)) {
                return std::nullopt;
            }
            fetch.attributes.push_back(Shader::Gcn::VertexAttribute{
                .dest_vgpr = fields[0],
                .num_elements = fields[1],
                .sgpr_base = fields[2],
                .dword_offset = fields[3],
                .instance_data = fields[4],
                .inst_offset = fields[5],
                .data_format = fields[6],
                .num_format = fields[7],
            });
        }
    }
    u32 num_buffers{};
    if (!GetState(in, num_buffers) || num_buffers > Shader::IR::NumParams) {
        return std::nullopt;
    }
    for (u32 i = 0; i < num_buffers; ++i) {
        std::pair<u32, AmdGpu::Buffer> buffer{};
        std::array<u32, 4> dwords{};
        if (!GetState(in, buffer.first) || !GetState(in, dwords)) {
            return std::nullopt;
        }
        static_assert(sizeof(AmdGpu::Buffer) == sizeof(dwords));
        std::memcpy(&buffer.second, dwords.data(), sizeof(dwords));
        state.vertex_buffers.push_back(buffer);
    }
    ColorBuffers color_buffers{};
    FillColorBuffers(*regs, color_buffers);
    FillRuntimeInfo(host, state.runtime_info, state.hw_stage, state.sw_stage, *regs, &cs_regs,
                    color_buffers, indirect, nullptr);
    return state;
}

/// A pixel shader's input semantics from its shader struct: after the stage registers, a count
/// byte (padded to a word), the input usage slots, then a u16 per input (semantic in the low
/// byte, flat shading in bit 10).
std::optional<std::vector<u16>> PsInputSemantics(std::span<const u8> header) {
    constexpr size_t CountOffset = 8 + 12 * sizeof(u32);
    if (header.size() < CountOffset + 4) {
        return std::nullopt;
    }
    u32 common;
    std::memcpy(&common, header.data(), sizeof(common));
    const u32 num_usage_slots = common >> 24;
    const u32 count = header[CountOffset];
    const size_t start = CountOffset + 4 + num_usage_slots * 4;
    // memcpy's pointers are declared nonnull, so with an empty vector the compiler may assume
    // count isn't zero and drop the vector's own empty checks: return before building one.
    if (count == 0 || start + count * sizeof(u16) > header.size()) {
        return std::nullopt;
    }
    std::vector<u16> semantics(count);
    std::memcpy(semantics.data(), header.data() + start, count * sizeof(u16));
    return semantics;
}

/// The fetch shader the game makes for a vertex shader, from the shader's struct: after the
/// seven stage registers, a word with the input count in its low byte, the input usage slots,
/// then 4 bytes per input (semantic, first VGPR, component count, reserved). The game's fetch
/// shader loads input i from the V# at dword 4i of the vertex buffer table that usage slot
/// type 0x17 points at.
std::optional<Shader::Gcn::FetchShaderData> GuessFetchShader(std::span<const u8> header) {
    constexpr size_t CountOffset = 8 + 7 * sizeof(u32);
    constexpr u8 UsagePtrVertexBufferTable = 0x17;
    if (header.size() < CountOffset + 4) {
        return std::nullopt;
    }
    u32 common;
    std::memcpy(&common, header.data(), sizeof(common));
    const u32 num_usage_slots = common >> 24;
    const u32 count = header[CountOffset];
    const size_t slots = CountOffset + 4;
    const size_t inputs = slots + num_usage_slots * 4;
    if (inputs + count * 4 > header.size()) {
        return std::nullopt;
    }
    Shader::Gcn::FetchShaderData fetch;
    if (count == 0) {
        return fetch;
    }
    u32 rsrc1;
    std::memcpy(&rsrc1, header.data() + 16, sizeof(rsrc1));
    const u32 num_vgprs = ((rsrc1 & 0x3f) + 1) * 4;
    std::optional<u8> table_sgpr;
    for (u32 i = 0; i < num_usage_slots; ++i) {
        if (header[slots + i * 4] == UsagePtrVertexBufferTable) {
            table_sgpr = header[slots + i * 4 + 2];
        }
    }
    // A struct read wrong would have the translation write past its registers, so anything
    // a real fetch shader couldn't do rules the guess out.
    if (!table_sgpr || *table_sgpr + 1 >= Shader::ShaderParams::NumShaderUserData ||
        count > Shader::IR::NumParams) {
        return std::nullopt;
    }
    for (u32 i = 0; i < count; ++i) {
        const u8* input = header.data() + inputs + i * 4;
        if (input[2] == 0 || input[2] > 4 || u32(input[1]) + input[2] > num_vgprs) {
            return std::nullopt;
        }
        fetch.attributes.push_back({
            .dest_vgpr = input[1],
            .num_elements = input[2],
            .sgpr_base = *table_sgpr,
            .dword_offset = static_cast<u8>(i * 4),
        });
    }
    // Only for telling an empty one apart; a load and the jump back per input.
    fetch.size = (count + 1) * 2 * sizeof(u32);
    return fetch;
}

std::optional<Shader::RuntimeInfo> GuessRuntimeInfo(const ShaderHost& host, HwStage hw_stage,
                                                    SwStage sw_stage, std::span<const u8> header,
                                                    int semantic_base) {
    // The stage registers start at PgmLo, two words into the shader struct.
    constexpr size_t RegsOffset = 8;
    const auto reg = [&](u32 i) {
        u32 value;
        std::memcpy(&value, header.data() + RegsOffset + i * sizeof(u32), sizeof(value));
        return value;
    };
    const auto settings = [&]<typename T>(const T&) {
        return std::bit_cast<T>(u64(reg(2)) | u64(reg(3)) << 32);
    };
    Shader::RuntimeInfo info{};
    info.Initialize(hw_stage, sw_stage);
    switch (hw_stage) {
    case HwStage::Fragment: {
        // PgmLo, PgmHi, Rsrc1, Rsrc2, ZFormat, ColFormat, InputEna, InputAddr, InControl,
        // BarycCntl, DbShaderControl, CbShaderMask.
        if (header.size() < RegsOffset + 12 * sizeof(u32)) {
            return std::nullopt;
        }
        const auto pgm = settings(AmdGpu::ShaderProgram{}.settings);
        info.props.fp_denorm_mode32 = pgm.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = pgm.fp_denorm_mode64;
        info.props.fp_round_mode32 = pgm.fp_round_mode32;
        info.props.fp_round_mode16_64 = pgm.fp_round_mode64;
        info.props.num_allocated_vgprs = (pgm.num_vgprs + 1) * 4;
        info.props.num_user_data = pgm.num_user_regs;
        info.props.num_input_vgprs = pgm.vgpr_comp_cnt;
        info.props.dx10_clamp = pgm.dx10_clamp;
        auto& fs = info.hw.fs;
        fs.z_export_format = static_cast<AmdGpu::ShaderExportFormat>(reg(4) & 0xf);
        const AmdGpu::ColorExportFormat col_format{reg(5)};
        fs.en_flags = std::bit_cast<AmdGpu::PsInput>(reg(6));
        fs.addr_flags = std::bit_cast<AmdGpu::PsInput>(reg(7));
        fs.num_inputs = reg(8) & 0x3f;
        if (fs.num_inputs > fs.inputs.size()) {
            return std::nullopt;
        }
        fs.front_face_all_bits = (reg(9) >> 24) & 1;
        const auto db = std::bit_cast<AmdGpu::DepthShaderControl>(reg(10));
        fs.depth_before_shader = db.depth_before_shader;
        const u8 stencil_export =
            db.stencil_op_val_export_enable | db.stencil_test_val_export_enable;
        fs.mrtz_mask = db.z_export_enable | (stencil_export << 1) | (db.mask_export_enable << 2) |
                       (db.coverage_to_mask_enable << 3);
        // Unknown without the draw: which vertex outputs the inputs read (from the semantics,
        // once draws have shown how they map), the render targets (assumed 8-bit unorm wherever
        // the shader exports), multisampling and blending (assumed off).
        const auto semantics = PsInputSemantics(header);
        int base = semantic_base;
        if (base < 0 && semantics && !semantics->empty()) {
            // Not learned yet: guess that the shader reads from the first vertex output.
            base = 0xff;
            for (const u16 semantic : *semantics) {
                base = std::min(base, int(semantic & 0xff));
            }
        }
        for (u32 i = 0; i < fs.num_inputs; ++i) {
            fs.inputs[i] = {.param_index = static_cast<u16>(i)};
            // param_index is 5 bits wide.
            if (semantics && semantics->size() == fs.num_inputs && base >= 0 &&
                int((*semantics)[i] & 0xff) >= base && int((*semantics)[i] & 0xff) - base < 32) {
                const u16 semantic = (*semantics)[i];
                fs.inputs[i].param_index = static_cast<u16>((semantic & 0xff) - base);
                fs.inputs[i].is_flat = (semantic >> 10) & 1;
            }
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; ++i) {
            const auto export_format = col_format.GetFormat(i);
            if (export_format == AmdGpu::ShaderExportFormat::Zero) {
                // Normalised like a draw's, whose unexported targets are left out the same way.
                fs.color_buffers[i] =
                    host.profile->fewer_shader_variants
                        ? ShaderColorBuffer(*host.profile, Shader::PsColorBuffer{})
                        : Shader::PsColorBuffer{};
                continue;
            }
            const auto target = UnboundTarget(export_format);
            fs.color_buffers[i] = host.profile->fewer_shader_variants
                                      ? ShaderColorBuffer(*host.profile, target)
                                      : target;
        }
        return info;
    }
    case HwStage::Vertex: {
        // PgmLo, PgmHi, Rsrc1, Rsrc2, VsOutConfig, PosFormat, PaClVsOutCntl.
        if (header.size() < RegsOffset + 7 * sizeof(u32) || sw_stage != SwStage::Vertex) {
            return std::nullopt;
        }
        const auto pgm = settings(AmdGpu::ShaderProgram{}.settings);
        info.props.fp_denorm_mode32 = pgm.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = pgm.fp_denorm_mode64;
        info.props.fp_round_mode32 = pgm.fp_round_mode32;
        info.props.fp_round_mode16_64 = pgm.fp_round_mode64;
        info.props.num_allocated_vgprs = (pgm.num_vgprs + 1) * 4;
        info.props.num_user_data = pgm.num_user_regs;
        info.props.num_input_vgprs = pgm.vgpr_comp_cnt;
        info.props.dx10_clamp = pgm.dx10_clamp;
        if (host.profile->fewer_shader_variants && host.vertex_input_dynamic_state) {
            // As BuildRuntimeInfo sets it.
            info.props.num_input_vgprs = 3;
        }
        // Unknown without the draw, assumed off: user clip planes, clipping disabled, a
        // [-w, w] depth range on drivers without depth clip control, instancing, indirect draws
        // and emulated rect and quad lists.
        info.hw.vs.num_outputs =
            MapOutputs(info.hw.vs.outputs, std::bit_cast<AmdGpu::VsOutputControl>(reg(6)));
        return info;
    }
    case HwStage::Compute: {
        // PgmLo, PgmHi, Rsrc1, Rsrc2, NumThreadX, NumThreadY, NumThreadZ.
        if (header.size() < RegsOffset + 7 * sizeof(u32)) {
            return std::nullopt;
        }
        const auto cs = settings(AmdGpu::ComputeProgram{}.settings);
        info.props.num_user_data = cs.num_user_regs;
        info.props.num_allocated_vgprs = cs.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs.fp_round_mode64;
        info.props.dx10_clamp = cs.dx10_clamp;
        info.hw.cs.workgroup_size = {reg(4) & 0xffff, reg(5) & 0xffff, reg(6) & 0xffff};
        const auto& size = info.hw.cs.workgroup_size;
        if (size[0] == 0 || size[1] == 0 || size[2] == 0 || size[0] * size[1] * size[2] > 1024) {
            return std::nullopt;
        }
        info.hw.cs.tgid_enable = {bool(cs.tgid_enable & 1), bool(cs.tgid_enable & 2),
                                  bool(cs.tgid_enable & 4)};
        info.hw.cs.shared_memory_size = cs.lds_dwords * 128 * 4;
        return info;
    }
    default:
        return std::nullopt;
    }
}

} // namespace Vulkan
