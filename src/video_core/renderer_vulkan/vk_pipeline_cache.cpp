// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ranges>
#include <fmt/ranges.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/scope_exit.h"
#include "common/soft_assert.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_resource_heaps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

/// Names the parts of two specializations of one shader that differ, so the log shows what each
/// new variant was compiled for. `a` is the new one: like the lookup, only resources it binds
/// are compared.
static std::vector<std::string> SpecDifferences(const Shader::StageSpecialization& a,
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

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
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
static Shader::PsColorBuffer UnboundTarget(AmdGpu::ShaderExportFormat export_format) {
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

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
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
        const auto* hull_info = infos[u32(SwStage::TessellationControl)];
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
        DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                   "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
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
            auto target = graphics_key.color_buffers[i];
            if (profile.runtime_target_swizzles && !info.hw.fs.dual_source_blending) {
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
            info.hw.fs.color_buffers[i] =
                profile.fewer_shader_variants ? ShaderColorBuffer(target) : target;
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
            profile.needs_clip_distance_emulation;
        break;
    }
    case HwStage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
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
        if (profile.fewer_shader_variants && instance.IsVertexInputDynamicState()) {
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
        info.sw.vs.vertex_sgpr_offset = draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.is_indirect = draw_indirect_params.is_indirect;
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
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
        .fast_shader_math = EmulatorSettings.IsFastShaderMath(),
        .support_bda = instance_.IsBufferDeviceAddressSupported(),
        .uses_descriptor_heap = instance_.UsesDescriptorHeap(),
        .fewer_shader_variants = EmulatorSettings.IsFewerShaderVariants(),
        .runtime_buffer_formats =
            instance_.UsesDescriptorHeap() && EmulatorSettings.IsFewerShaderVariants(),
        .runtime_buffer_strides =
            instance_.UsesDescriptorHeap() && EmulatorSettings.IsFewerShaderVariants(),
        .runtime_storage_swizzles =
            instance_.UsesDescriptorHeap() && EmulatorSettings.IsFewerShaderVariants(),
        .runtime_target_swizzles =
            instance_.UsesDescriptorHeap() && EmulatorSettings.IsFewerShaderVariants(),
        // They make the driver's vertex shader compiles about a fifth slower, which only pays
        // when it lets guessed shaders compiled early serve their draws.
        .runtime_vertex_formats = instance_.UsesDescriptorHeap() &&
                                  EmulatorSettings.IsFewerShaderVariants() &&
                                  EmulatorSettings.GetShaderPrecompileThreads() != 0,
    };
    // A quarter of the cores: the game's own threads and the GPU thread need the rest.
    const u32 compile_threads = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
    compiler = std::make_unique<PipelineCompiler>(compile_threads);
    if (instance.UsesShaderObjects()) {
        shader_objects = std::make_unique<ShaderObjects>(instance);
    }
    if (EmulatorSettings.IsShaderPrescan()) {
        shader_scanner = std::make_unique<VideoCore::ShaderScanner>();
        // Temporary, for measuring guessed translation.
        const char* check = std::getenv("SHADPS4_GUESS_CHECK");
        guess_check = check && check[0] == '1';
        // Measurement: translates the shaders saved by earlier runs, to time translation alone.
        if (const char* bench = std::getenv("SHADPS4_TRANSLATE_BENCH")) {
            const u32 rounds = std::max<u32>(1, std::strtoul(bench, nullptr, 10));
            bench_thread = std::jthread([this, rounds] { RunTranslateBench(rounds); });
        }
    }
    LOG_INFO(Render_Vulkan, "{} pipeline compile threads", compile_threads);
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    if (const u32 threads = EmulatorSettings.GetShaderPrecompileThreads(); threads != 0) {
        if (!shader_scanner || !shader_scanner->Running() || !profile.uses_descriptor_heap) {
            LOG_WARNING(Render_Vulkan, "Compiling found shaders early needs the shader scan and "
                                       "the descriptor heap; it's off");
        } else {
            StartPrecompile(threads);
        }
    }
}

/// Stands for a stage that's never compiled early, such as a domain shader on the vertex stage.
constexpr auto UnguessedStage = static_cast<HwStage>(0xfe);

/// The stage a first use counts as for early compiling: its hardware stage, unless the hardware
/// stage runs something other than its usual kind of shader.
static HwStage PrecompileStageOf(HwStage hw_stage, SwStage sw_stage) {
    const bool usual = (hw_stage == HwStage::Vertex && sw_stage == SwStage::Vertex) ||
                       (hw_stage == HwStage::Fragment && sw_stage == SwStage::Fragment) ||
                       (hw_stage == HwStage::Compute && sw_stage == SwStage::Compute);
    return usual ? hw_stage : UnguessedStage;
}

/// Whether a translation from a guessed stage came out as something that stage can be. A
/// guess of the wrong stage mustn't reach the backend's checks, which end the emulator: shared
/// memory on a graphics stage means the shader is really a hull, local or export shader.
static bool GuessTranslated(const Shader::Info& info, HwStage hw_stage) {
    return !info.translation_failed &&
           (hw_stage == HwStage::Compute || static_cast<u32>(info.shared_types) == 0);
}

/// The program user data a prepared program keeps until its first use fills in the draw's.
static constexpr std::array<u32, Shader::ShaderParams::NumShaderUserData> NoUserData{};

/// The stages whose register state a shader's struct gives.
static bool PrecompilesStage(HwStage stage) {
    return stage == HwStage::Fragment || stage == HwStage::Compute || stage == HwStage::Vertex;
}

PipelineCache::~PipelineCache() {
    StopPrecompile();
    if (bench_thread.joinable()) {
        bench_thread.join();
    }
}

void PipelineCache::RunTranslateBench(u32 rounds) {
    Common::SetCurrentThreadName("shadPS4:TranslateBench");
    shader_scanner->WaitUntilLoaded();
    const auto hashes = shader_scanner->KnownWithStage();
    LOG_INFO(Render_Vulkan, "Translation bench: {} saved shaders with a known stage, {} rounds",
             hashes.size(), rounds);
    struct Timed {
        u64 ns;
        u64 hash;
        HwStage stage;
        size_t dwords;
    };
    std::vector<Timed> timed;
    tsl::robin_map<u64, size_t> timed_index;
    std::array<u64, 7> stage_ns{};
    std::array<u32, 7> stage_count{};
    Shader::Pools bench_pools;
    Shader::TranslateTimes phases{};
    u64 emit_ns = 0;
    u64 total_ns = 0;
    u64 dwords = 0;
    u32 translated = 0;
    u32 unguessable = 0;
    static constexpr std::array<u32, Shader::ShaderParams::NumShaderUserData> NoUserData{};
    for (u32 round = 0; round < rounds; ++round) {
        for (const u64 hash : hashes) {
            const auto found = shader_scanner->Find(hash);
            if (!found) {
                continue;
            }
            const auto hw_stage = static_cast<HwStage>(found->stage);
            if (!PrecompilesStage(hw_stage)) {
                continue;
            }
            const auto sw_stage = hw_stage == HwStage::Compute  ? SwStage::Compute
                                  : hw_stage == HwStage::Vertex ? SwStage::Vertex
                                                                : SwStage::Fragment;
            auto runtime_info = GuessRuntimeInfo(hw_stage, sw_stage, found->header);
            std::optional<Shader::Gcn::FetchShaderData> fetch;
            if (hw_stage == HwStage::Vertex) {
                fetch = GuessFetchShader(found->header);
            }
            if (!runtime_info || (hw_stage == HwStage::Vertex && !fetch)) {
                unguessable += round == 0 ? 1 : 0;
                continue;
            }
            const Shader::ShaderParams params{
                .user_data = NoUserData,
                .code = found->code,
                .hash = hash,
            };
            Shader::Info info(hw_stage, sw_stage, params);
            info.guess_state = true;
            info.guessed_fetch = fetch ? &*fetch : nullptr;
            Shader::Backend::Bindings binding{};
            const auto before = Shader::ThreadTranslateTimes();
            const u64 start = VideoCore::PerfStats::NowNs();
            u64 emit_start = 0;
            u64 end = 0;
            try {
                const Common::SoftAsserts soft_asserts;
                const auto ir_program = Shader::TranslateProgram(found->code, bench_pools, info,
                                                                 *runtime_info, profile);
                if (!GuessTranslated(info, hw_stage)) {
                    throw Common::SoftAssertFailure{};
                }
                emit_start = VideoCore::PerfStats::NowNs();
                const auto spv =
                    Shader::Backend::SPIRV::EmitSPIRV(profile, *runtime_info, ir_program, binding);
                end = VideoCore::PerfStats::NowNs();
            } catch (const Common::SoftAssertFailure&) {
                unguessable += round == 0 ? 1 : 0;
                continue;
            }
            const auto& after = Shader::ThreadTranslateTimes();
            for (size_t i = 0; i < phases.ns.size(); ++i) {
                phases.ns[i] += after.ns[i] - before.ns[i];
            }
            emit_ns += end - emit_start;
            total_ns += end - start;
            dwords += found->code.size();
            ++translated;
            const auto stage_index = static_cast<size_t>(hw_stage);
            stage_ns[stage_index] += end - start;
            ++stage_count[stage_index];
            // Each shader's best round, for the slowest list.
            if (const auto [it, added] = timed_index.try_emplace(hash, timed.size()); added) {
                timed.push_back({end - start, hash, hw_stage, found->code.size()});
            } else {
                timed[it->second].ns = std::min(timed[it->second].ns, end - start);
            }
        }
    }
    std::vector<std::pair<u64, std::string_view>> parts;
    for (size_t i = 0; i < phases.ns.size(); ++i) {
        parts.emplace_back(phases.ns[i],
                           Shader::TranslatePhaseName(static_cast<Shader::TranslatePhase>(i)));
    }
    parts.emplace_back(emit_ns, "SPIR-V emission");
    std::ranges::sort(parts, std::greater{});
    std::string list;
    for (const auto& [ns, name] : parts) {
        list += fmt::format("{}{} {:.1f} ms ({:.0f}%)", list.empty() ? "" : ", ", name, ns / 1e6,
                            100.0 * double(ns) / double(std::max<u64>(total_ns, 1)));
    }
    std::string stages;
    for (size_t i = 0; i < stage_ns.size(); ++i) {
        if (stage_count[i] != 0) {
            stages += fmt::format("{}{} {} in {:.1f} ms", stages.empty() ? "" : ", ",
                                  static_cast<HwStage>(i), stage_count[i], stage_ns[i] / 1e6);
        }
    }
    LOG_INFO(Render_Vulkan,
             "Translation bench: {} translations in {:.1f} ms ({:.1f} us per shader, {:.0f} us "
             "per 1000 code dwords), {} without a usable struct; by stage: {}; by phase: {}",
             translated, total_ns / 1e6, translated != 0 ? total_ns / 1e3 / translated : 0.0,
             dwords != 0 ? total_ns / 1e3 / (dwords / 1e3) : 0.0, unguessable, stages, list);
    std::ranges::sort(timed, std::greater{}, &Timed::ns);
    for (size_t i = 0; i < std::min<size_t>(timed.size(), 10); ++i) {
        LOG_INFO(Render_Vulkan,
                 "Translation bench: slowest {}: {} shader {:#x}, {} dwords, {:.2f} ms", i + 1,
                 timed[i].stage, timed[i].hash, timed[i].dwords, timed[i].ns / 1e6);
    }
}

void PipelineCache::TickCompileStats() {
    constexpr u64 IntervalNs = 5'000'000'000ULL;
    const u64 now = VideoCore::PerfStats::NowNs();
    if (compile_stats.last_log_ns == 0) {
        compile_stats.last_log_ns = now;
        ab_switched_ns = now;
        VideoCore::PerfStats::frames.Take();
        return;
    }
    // The A/B timer flips sync and async creation in one run; each switch ends a stats window
    // so that no line mixes the two.
    const u64 ab_ns = u64(EmulatorSettings.GetAsyncPipelinesAbSeconds()) * 1'000'000'000;
    if (ab_ns != 0 && now - ab_switched_ns >= ab_ns) {
        FlushCompileStats(now);
        ab_async = !ab_async;
        ab_switched_ns = now;
        LOG_INFO(Render_Vulkan, "Async pipelines A/B: now {}", AsyncModeName(GetAsyncMode()));
        return;
    }
    if (now - compile_stats.last_log_ns >= IntervalNs) {
        FlushCompileStats(now);
    }
}

void PipelineCache::FlushCompileStats(u64 now_ns) {
    auto& stats = compile_stats;
    auto& bg = background_stats;
    const auto frames = VideoCore::PerfStats::frames.Take();
    const double seconds = double(now_ns - stats.last_log_ns) / 1e9;
    const u64 bg_created = bg.created.exchange(0, std::memory_order_relaxed);
    const u64 bg_driver_ns = bg.driver_ns.exchange(0, std::memory_order_relaxed);
    const u64 bg_worst_driver_ns = bg.worst_driver_ns.exchange(0, std::memory_order_relaxed);
    const u64 bg_worst_latency_ns = bg.worst_latency_ns.exchange(0, std::memory_order_relaxed);
    if (frames.count != 0 || stats.graphics != 0 || stats.compute != 0 || bg_created != 0) {
        LOG_INFO(Render_Vulkan,
                 "Frames over {:.1f} s: {} ({:.1f} FPS), {} over 33 ms, worst {:.1f} ms, "
                 "99th percentile {:.1f} ms, 99.9th {:.1f} ms; {} stutters ({} over 50 ms, {} "
                 "over 100 ms) stalling {:.1f} ms, since boot {} stalling {:.1f} ms. "
                 "Compiled {} graphics + {} compute pipelines from {} shaders ({} permutations, "
                 "{} only for binding offsets): recompile "
                 "{:.1f} ms, driver {:.1f} ms, worst pipeline {:.1f} ms. "
                 "Async {}: {} sent to workers, {} created there (driver {:.1f} ms, worst "
                 "{:.1f} ms, ready at worst {:.1f} ms after the draw), {} draws skipped, "
                 "{} draws waited for a new target ({} ran on the GPU thread), waited {:.1f} ms, "
                 "{} pending",
                 seconds, frames.count, frames.count / seconds, frames.hitches,
                 frames.worst_ns / 1e6, frames.p99_ns / 1e6, frames.p999_ns / 1e6, frames.stutters,
                 frames.over_50, frames.over_100, frames.stall_ns / 1e6, frames.total_stutters,
                 frames.total_stall_ns / 1e6, stats.graphics, stats.compute, stats.shaders,
                 stats.permutations, stats.binding_permutations, stats.recompile_ns / 1e6,
                 stats.driver_ns / 1e6, stats.worst_ns / 1e6, AsyncModeName(GetAsyncMode()),
                 stats.background, bg_created, bg_driver_ns / 1e6, bg_worst_driver_ns / 1e6,
                 bg_worst_latency_ns / 1e6, stats.skipped_draws, stats.waited_draws, stats.ran_now,
                 stats.waited_ns / 1e6, compiler->Pending());
    }
    if (stats.shaders != 0) {
        // Where the GPU thread's recompile time went, largest first.
        std::vector<std::pair<u64, std::string_view>> parts;
        u64 translate_ns = 0;
        for (size_t i = 0; i < stats.translate.ns.size(); ++i) {
            translate_ns += stats.translate.ns[i];
            parts.emplace_back(stats.translate.ns[i],
                               Shader::TranslatePhaseName(static_cast<Shader::TranslatePhase>(i)));
        }
        parts.emplace_back(stats.emit_ns, "SPIR-V emission");
        parts.emplace_back(stats.module_ns, "shader modules");
        const u64 known_ns = translate_ns + stats.emit_ns + stats.module_ns;
        parts.emplace_back(stats.recompile_ns > known_ns ? stats.recompile_ns - known_ns : 0,
                           "the rest");
        std::ranges::sort(parts, std::greater{});
        std::string list;
        for (const auto& [ns, name] : parts) {
            if (ns != 0) {
                list += fmt::format("{}{} {:.1f} ms", list.empty() ? "" : ", ", name, ns / 1e6);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "Recompile time over {:.1f} s: {:.1f} ms for {} shaders ({:.0f} us per 1000 "
                 "code dwords): {}",
                 seconds, stats.recompile_ns / 1e6, stats.shaders,
                 stats.code_dwords != 0 ? stats.recompile_ns / 1e3 / (stats.code_dwords / 1e3)
                                        : 0.0,
                 list);
    }
    if (stats.prefetched != 0 || stats.prefetch_used != 0 || stats.waited_dispatches != 0) {
        LOG_INFO(Render_Vulkan,
                 "Look-ahead over {:.1f} s: {} pipelines made ahead of their draw, {} found later "
                 "by a draw, {} dispatches waited",
                 seconds, stats.prefetched, stats.prefetch_used, stats.waited_dispatches);
    }
    if (shader_objects) {
        const auto objects = shader_objects->TakeCounts();
        if (objects.created != 0 || objects.reused != 0) {
            LOG_INFO(Render_Vulkan, "Shader objects over {:.1f} s: {} created, {} reused", seconds,
                     objects.created, objects.reused);
        }
    }
    if (shader_scanner) {
        const auto scan = shader_scanner->TakeCounts();
        if (scan.first_uses != 0 || scan.scanned_mb != 0) {
            LOG_INFO(Render_Vulkan,
                     "Shader scan over {:.1f} s: {} of {} new programs found ahead (median "
                     "{:.1f} s ahead), {} known ({} with stage registers); read {} MB in {:.0f} ms",
                     seconds, scan.found_ahead, scan.first_uses, scan.median_lead_s, scan.known,
                     scan.with_registers, scan.scanned_mb, scan.scan_ms);
        }
    }
    LogGuessStats();
    if (precompile.active) {
        std::scoped_lock lock{precompile.mutex};
        if (precompile.translated != 0 || precompile.served != 0 || precompile.missed != 0) {
            LOG_INFO(Render_Vulkan,
                     "Compiled early over {:.1f} s: {} shaders ({} shader objects, {} compute "
                     "pipelines) in {:.0f} ms of worker time; first uses: {} served, {} needed "
                     "another variant; {} came too late; {} queued; "
                     "{} dropped as the wrong stage; {} first uses waited {:.1f} ms for the "
                     "worker compiling them; {} queued were used before a worker got to them; "
                     "{} compiled from recorded draw states ({} variants)",
                     seconds, precompile.translated, precompile.shader_objects,
                     precompile.compute_pipelines, precompile.worker_ns / 1e6, precompile.served,
                     precompile.missed, precompile.duplicate, precompile.queue.size(),
                     precompile.rejected, precompile.joined, precompile.joined_ns / 1e6,
                     precompile.skipped_used, precompile.replayed, precompile.replayed_variants);
            precompile.replayed = precompile.replayed_variants = 0;
            precompile.skipped_used = 0;
            precompile.joined = 0;
            precompile.joined_ns = 0;
            precompile.translated = precompile.shader_objects = precompile.compute_pipelines = 0;
            precompile.served = precompile.missed = precompile.duplicate = 0;
            precompile.rejected = 0;
            precompile.worker_ns = 0;
        }
    }
    if (const auto gpu = scheduler.TakeGpuTime(); gpu.submits != 0) {
        LOG_INFO(Render_Vulkan,
                 "GPU over {:.1f} s: busy {:.1f} ms ({:.1f} ms per frame over {} frames), {} "
                 "submits, longest {:.2f} ms",
                 seconds, gpu.busy_ns / 1e6,
                 frames.count != 0 ? gpu.busy_ns / 1e6 / frames.count : 0.0, frames.count,
                 gpu.submits, gpu.worst_ns / 1e6);
    }
    const u64 probes = bg.probes.exchange(0, std::memory_order_relaxed);
    const u64 probe_optimized_ns = bg.probe_optimized_ns.exchange(0, std::memory_order_relaxed);
    const u64 probe_unoptimized_ns = bg.probe_unoptimized_ns.exchange(0, std::memory_order_relaxed);
    if (probes != 0) {
        LOG_INFO(Render_Vulkan,
                 "Pipeline probes over {:.1f} s: {} pipelines, optimized {:.1f} ms, "
                 "unoptimized {:.1f} ms ({:.0f}%)",
                 seconds, probes, probe_optimized_ns / 1e6, probe_unoptimized_ns / 1e6,
                 100.0 * double(probe_unoptimized_ns) /
                     double(std::max<u64>(probe_optimized_ns, 1)));
    }
    stats = CompileStats{.last_log_ns = now_ns};
}

PipelineCache::AsyncMode PipelineCache::GetAsyncMode() const {
    const u32 setting = EmulatorSettings.GetAsyncPipelines();
    const auto mode = setting <= 3 ? static_cast<AsyncMode>(setting) : AsyncMode::Off;
    const auto chosen = [this](AsyncMode m) {
        // Shader objects have no unoptimized version to start with, so tiered is sync there.
        return shader_objects && m == AsyncMode::Tiered ? AsyncMode::Off : m;
    };
    if (EmulatorSettings.GetAsyncPipelinesAbSeconds() != 0) {
        // Flips between off and the chosen mode (background skipping if none is chosen).
        return ab_async ? chosen(mode == AsyncMode::Off ? AsyncMode::Skip : mode) : AsyncMode::Off;
    }
    return chosen(mode);
}

const char* PipelineCache::AsyncModeName(AsyncMode mode) {
    switch (mode) {
    case AsyncMode::Skip:
        return "skip";
    case AsyncMode::Tiered:
        return "tiered";
    case AsyncMode::SkipAll:
        return "skip all";
    default:
        return "off";
    }
}

static void AtomicMax(std::atomic<u64>& value, u64 candidate) {
    u64 current = value.load(std::memory_order_relaxed);
    while (candidate > current &&
           !value.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
    }
}

PipelineCompileOptions PipelineCache::MakeCompileOptions(u64 pipeline_hash, u64 recompile_ns,
                                                         AsyncMode mode) {
    PipelineCompileOptions options{};
    if (mode != AsyncMode::Off) {
        options.compiler = compiler.get();
        options.tiered = mode == AsyncMode::Tiered;
    }
    if (EmulatorSettings.IsPipelineCompileProbe() && mode == AsyncMode::Off) {
        options.prober = compiler.get();
    }
    auto* bg = &background_stats;
    const u64 asked_ns = VideoCore::PerfStats::NowNs();
    const auto background = [bg, asked_ns](u64 driver_ns) {
        const u64 latency_ns = VideoCore::PerfStats::NowNs() - asked_ns;
        bg->created.fetch_add(1, std::memory_order_relaxed);
        bg->driver_ns.fetch_add(driver_ns, std::memory_order_relaxed);
        AtomicMax(bg->worst_driver_ns, driver_ns);
        AtomicMax(bg->worst_latency_ns, latency_ns);
        return latency_ns;
    };
    if (mode == AsyncMode::Skip || mode == AsyncMode::SkipAll) {
        options.on_created = [background, pipeline_hash, recompile_ns](u64 driver_ns) {
            const u64 latency_ns = background(driver_ns);
            LOG_INFO(Render_Vulkan,
                     "Graphics pipeline {:#x}: recompile {:.2f} ms, driver {:.2f} ms, in the "
                     "background, ready {:.2f} ms after the draw",
                     pipeline_hash, recompile_ns / 1e6, driver_ns / 1e6, latency_ns / 1e6);
        };
    } else {
        options.on_created = [pipeline_hash, recompile_ns, mode](u64 driver_ns) {
            LOG_INFO(Render_Vulkan,
                     "Graphics pipeline {:#x}: recompile {:.2f} ms, driver {:.2f} ms{}",
                     pipeline_hash, recompile_ns / 1e6, driver_ns / 1e6,
                     mode == AsyncMode::Tiered ? ", unoptimized" : "");
        };
    }
    if (mode == AsyncMode::Tiered) {
        options.on_upgraded = [background, pipeline_hash](u64 driver_ns) {
            const u64 latency_ns = background(driver_ns);
            LOG_INFO(Render_Vulkan,
                     "Graphics pipeline {:#x}: optimized in the background, driver {:.2f} ms, "
                     "replaced the unoptimized one {:.2f} ms after the draw",
                     pipeline_hash, driver_ns / 1e6, latency_ns / 1e6);
        };
    }
    if (options.prober) {
        options.on_probed = [bg, pipeline_hash](u64 optimized_ns, u64 unoptimized_ns) {
            bg->probes.fetch_add(1, std::memory_order_relaxed);
            bg->probe_optimized_ns.fetch_add(optimized_ns, std::memory_order_relaxed);
            bg->probe_unoptimized_ns.fetch_add(unoptimized_ns, std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan,
                     "Graphics pipeline {:#x} probe: optimized {:.2f} ms, unoptimized {:.2f} ms",
                     pipeline_hash, optimized_ns / 1e6, unoptimized_ns / 1e6);
        };
    }
    return options;
}

void PipelineCache::WaitUntilReady(const GraphicsPipeline& pipeline) {
    const u64 start = VideoCore::PerfStats::NowNs();
    // Its job may sit behind the ones for skipped draws, so run it here if it hasn't started.
    if (pipeline.CompileTicket() != 0 && compiler->RunNow(pipeline.CompileTicket())) {
        ++compile_stats.ran_now;
    }
    compiler->WaitFor([&pipeline] { return pipeline.IsReady(); }, std::chrono::hours{1});
    const u64 waited = VideoCore::PerfStats::NowNs() - start;
    compile_stats.waited_ns += waited;
    VideoCore::PerfStats::frames.AddShaderTime(waited);
    compile_stats.worst_ns = std::max(compile_stats.worst_ns, waited);
    ++compile_stats.waited_draws;
}

u64 PipelineCache::WaitForPipeline(const GraphicsPipeline& pipeline) {
    // Refills at budget per 60 Hz frame, up to one budget, so a burst of new pipelines can't
    // stall a frame by more than about the budget.
    constexpr u64 FrameNs = 16'666'667;
    const u64 budget_ns = u64(EmulatorSettings.GetAsyncPipelineWaitMs()) * 1'000'000;
    if (budget_ns == 0) {
        return 0;
    }
    const u64 now = VideoCore::PerfStats::NowNs();
    const u64 elapsed = std::min<u64>(now - wait_credit_ns_at, 1'000'000'000);
    wait_credit_ns = std::min(budget_ns, wait_credit_ns + elapsed * budget_ns / FrameNs);
    wait_credit_ns_at = now;
    if (wait_credit_ns == 0) {
        return 0;
    }
    compiler->WaitFor([&pipeline] { return pipeline.IsReady(); },
                      std::chrono::nanoseconds{wait_credit_ns});
    const u64 waited = VideoCore::PerfStats::NowNs() - now;
    wait_credit_ns -= std::min(wait_credit_ns, waited);
    compile_stats.waited_ns += waited;
    VideoCore::PerfStats::frames.AddShaderTime(waited);
    return waited;
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    return LookUpGraphicsPipeline(params, false);
}

bool PipelineCache::PrefetchGraphicsPipeline() {
    const auto mode = GetAsyncMode();
    if (mode != AsyncMode::Skip && mode != AsyncMode::SkipAll) {
        return false;
    }
    const size_t before = graphics_pipelines.size();
    LookUpGraphicsPipeline({}, true);
    return graphics_pipelines.size() != before;
}

const GraphicsPipeline* PipelineCache::LookUpGraphicsPipeline(const DrawIndirectParams params,
                                                              bool prefetch) {
    draw_indirect_params = params;
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (!is_new && !prefetch && !prefetched.empty() && prefetched.erase(it->second.get()) != 0) {
        ++compile_stats.prefetch_used;
    }
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        // Stage variants in logical stage order (fs, tcs, tes, vs, gs), 0 when unused, so a log
        // shows how many pipelines share each shader.
        const auto& h = graphics_key.stage_hashes;
        LOG_INFO(Render_Vulkan,
                 "Compiling graphics pipeline {:#x} (stages {:#x} {:#x} {:#x} {:#x} {:#x})",
                 pipeline_hash, h[0], h[1], h[2], h[3], h[4]);

        GraphicsPipeline::SerializationSupport sdata{};
        const u64 recompile_ns = std::exchange(pending_recompile_ns, 0);
        const auto mode = GetAsyncMode();
        auto options = MakeCompileOptions(pipeline_hash, recompile_ns, mode);
        options.shader_objects = shader_objects.get();
        const u64 start = VideoCore::PerfStats::NowNs();
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false, std::move(options));
        // When skipping this is only the setup, as the workers time the driver.
        const u64 gpu_thread_ns = VideoCore::PerfStats::NowNs() - start;
        ++compile_stats.graphics;
        if (mode != AsyncMode::Off) {
            ++compile_stats.background;
        }
        const bool skips = mode == AsyncMode::Skip || mode == AsyncMode::SkipAll;
        if (!skips) {
            compile_stats.driver_ns += gpu_thread_ns;
            VideoCore::PerfStats::frames.AddShaderTime(gpu_thread_ns);
        }
        // A prefetch never waits: the draw it was made for does that if it must.
        const u64 waited_ns = skips && !prefetch ? WaitForPipeline(*it.value()) : 0;
        if (prefetch) {
            ++compile_stats.prefetched;
            prefetched.insert(it.value().get());
        }
        compile_stats.worst_ns =
            std::max(compile_stats.worst_ns, recompile_ns + gpu_thread_ns + waited_ns);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    const ComputePipeline* pipeline = LookUpComputePipeline(false);
    if (pipeline && !pipeline->IsReady()) {
        // Made ahead on the workers; a dispatch can't be skipped, so it waits.
        const u64 start = VideoCore::PerfStats::NowNs();
        if (pipeline->CompileTicket() != 0 && compiler->RunNow(pipeline->CompileTicket())) {
            ++compile_stats.ran_now;
        }
        compiler->WaitFor([pipeline] { return pipeline->IsReady(); }, std::chrono::hours{1});
        const u64 waited = VideoCore::PerfStats::NowNs() - start;
        compile_stats.waited_ns += waited;
        VideoCore::PerfStats::frames.AddShaderTime(waited);
        compile_stats.worst_ns = std::max(compile_stats.worst_ns, waited);
        ++compile_stats.waited_dispatches;
    }
    return pipeline;
}

bool PipelineCache::PrefetchComputePipeline() {
    const auto mode = GetAsyncMode();
    if (mode != AsyncMode::Skip && mode != AsyncMode::SkipAll) {
        return false;
    }
    const size_t before = compute_pipelines.size();
    LookUpComputePipeline(true);
    return compute_pipelines.size() != before;
}

const ComputePipeline* PipelineCache::LookUpComputePipeline(bool prefetch) {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (!is_new && !prefetch && !prefetched.empty() && prefetched.erase(it->second.get()) != 0) {
        ++compile_stats.prefetch_used;
    }
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        const u64 start = VideoCore::PerfStats::NowNs();
        it.value() = std::make_unique<ComputePipeline>(
            instance, scheduler, desc_heap, profile, *pipeline_cache, compute_key, *infos[0],
            modules[0], sdata, false, prefetch ? compiler.get() : nullptr);
        const u64 gpu_thread_ns = VideoCore::PerfStats::NowNs() - start;
        const u64 recompile_ns = std::exchange(pending_recompile_ns, 0);
        ++compile_stats.compute;
        if (prefetch) {
            ++compile_stats.prefetched;
            prefetched.insert(it.value().get());
            LOG_INFO(Render_Vulkan,
                     "Compute pipeline {:#x}: recompile {:.2f} ms, ahead on the "
                     "workers",
                     pipeline_hash, recompile_ns / 1e6);
        } else {
            new_compute_pipeline = true;
            compile_stats.driver_ns += gpu_thread_ns;
            VideoCore::PerfStats::frames.AddShaderTime(gpu_thread_ns);
            compile_stats.worst_ns = std::max(compile_stats.worst_ns, recompile_ns + gpu_thread_ns);
            LOG_INFO(Render_Vulkan, "Compute pipeline {:#x}: recompile {:.2f} ms, driver {:.2f} ms",
                     pipeline_hash, recompile_ns / 1e6, gpu_thread_ns / 1e6);
        }
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
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

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = nullptr;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        if (profile.uses_descriptor_heap) {
            // Each stage maps its own bindings onto the heap, so its numbers start at 0.
            binding = {};
        }
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(HwStage::Fragment, SwStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

std::pair<vk::ShaderModule, u64> PipelineCache::CompileModule(Shader::Info& info,
                                                              Shader::RuntimeInfo& runtime_info,
                                                              const std::span<const u32>& code,
                                                              size_t perm_idx,
                                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");
    const u64 start = VideoCore::PerfStats::NowNs();
    SCOPE_EXIT {
        // Translation plus shader module creation, which drivers mostly defer to the pipeline.
        const u64 elapsed = VideoCore::PerfStats::NowNs() - start;
        pending_recompile_ns += elapsed;
        compile_stats.recompile_ns += elapsed;
        VideoCore::PerfStats::frames.AddShaderTime(elapsed);
        ++compile_stats.shaders;
    };

    const auto translate_before = Shader::ThreadTranslateTimes();
    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    const auto& translate_after = Shader::ThreadTranslateTimes();
    for (size_t i = 0; i < translate_after.ns.size(); ++i) {
        compile_stats.translate.ns[i] += translate_after.ns[i] - translate_before.ns[i];
    }
    compile_stats.code_dwords += code.size();
    const u64 emit_start = VideoCore::PerfStats::NowNs();
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    const u64 module_start = VideoCore::PerfStats::NowNs();
    compile_stats.emit_ns += module_start - emit_start;
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    vk::ShaderModule module;

    // Only looked up when patching is on: it's two file system checks per shader.
    auto patch = EmulatorSettings.IsPatchShaders()
                     ? GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv")
                     : std::nullopt;
    const bool is_patched = patch.has_value();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }
    compile_stats.module_ns += VideoCore::PerfStats::NowNs() - module_start;

    if (shader_objects) {
        shader_objects->Register(module, is_patched ? std::span<const u32>{*patch} : spv);
    }
    // Identified by what it compiles to, not by the order the variants were made in.
    const u64 hash = HashCombine(info.pgm_hash, XXH3_64bits(spv.data(), spv.size() * sizeof(u32)));
    RegisterShaderBinary(std::move(spv), hash);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return {module, hash};
}

/// The parts of a render target that change a fragment shader's code. The output type follows
/// the number class, not the exact format, and a swizzle's unused components map to
/// themselves, since the target drops them anyway. Keeping only these lets one shader serve
/// targets of different formats.
Shader::PsColorBuffer PipelineCache::ShaderColorBuffer(const Shader::PsColorBuffer& target) const {
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

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    if (precompile.active) {
        JoinPrecompile(params.hash);
        TakePrecompiled();
    }
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        // A domain shader runs on the hardware vertex stage but isn't a vertex shader: it's
        // noted as a stage that isn't compiled early.
        const auto noted_stage = PrecompileStageOf(hw_stage, sw_stage);
        if (shader_scanner) {
            shader_scanner->NoteFirstUse(params.hash, static_cast<u8>(noted_stage));
        }
        if (precompile.active) {
            NotePrecompileType(params.hash, noted_stage);
        }
        if (shader_scanner && hw_stage == HwStage::Fragment) {
            LearnSemanticBase(params.hash, runtime_info);
        }
        it_pgm.value() = std::make_unique<Program>(hw_stage, sw_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto [module, perm_hash] =
            CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        RecordDrawState(program->info, runtime_info, spec.fetch_shader_data);
        if (guess_check && shader_scanner && shader_scanner->Find(params.hash)) {
            CheckGuess(hw_stage, sw_stage, params, runtime_info, start, spec, perm_hash);
        }

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec), perm_hash);
        if (auto& fetch = program->modules[0].spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
        }
        return std::make_tuple(&program->info, module, perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    size_t perm_idx = program->modules.size();
    u64 perm_hash{};

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (program->prepared) {
        NotePreparedUse(*program, params.hash, it != program->modules.end(), spec,
                        it != program->modules.end() ? it->hash : 0);
        RecordDrawState(info, runtime_info, spec.fetch_shader_data);
    }
    if (it == program->modules.end()) {
        ++compile_stats.permutations;
        // Counts variants that only exist because an earlier stage moved the bindings.
        const bool binding_only = std::ranges::any_of(program->modules, [&](const auto& m) {
            auto moved = spec;
            moved.start = m.spec.start;
            return moved == m.spec;
        });
        if (binding_only) {
            ++compile_stats.binding_permutations;
        }
        // Compares with the closest existing variant, to show what this one is compiled for.
        std::vector<std::string> closest;
        for (const auto& m : program->modules) {
            auto diff = SpecDifferences(spec, m.spec);
            if (closest.empty() || diff.size() < closest.size()) {
                closest = std::move(diff);
            }
        }
        LOG_INFO(Render_Vulkan, "Variant {} of {} shader {:#x} differs in: {}", perm_idx, hw_stage,
                 params.hash, fmt::join(closest, ", "));
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        std::tie(module, perm_hash) =
            CompileModule(new_info, runtime_info, params.code, perm_idx, binding);
        RecordDrawState(new_info, runtime_info, spec.fetch_shader_data);

        if (const auto guessed = guessed_hashes.find(params.hash);
            guessed != guessed_hashes.end()) {
            auto& later = guessed.value();
            if (later.resources == perm_hash || later.full == perm_hash) {
                LOG_INFO(Render_Vulkan, "Guess for {} shader {:#x} matches variant {}{}", hw_stage,
                         params.hash, perm_idx,
                         later.full == perm_hash ? " with struct registers" : "");
                guess_stats.changed = true;
            }
            if (later.resources == perm_hash) {
                ++guess_stats.matched_later;
                later.resources = 0;
            }
            if (later.full == perm_hash) {
                ++guess_stats.full_matched_later;
                later.full = 0;
            }
            if (later.resources == 0 && later.full == 0) {
                guessed_hashes.erase(guessed);
            }
        }

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec), perm_hash);
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = it->hash;
    }
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    return std::make_tuple(&program->info, module, perm_hash);
}

void PipelineCache::StartPrecompile(u32 num_threads) {
    // The binary info's type is the program type the shader was compiled as. Seen in every
    // game so far: 0 pixel, 1 vertex (on the vertex stage), 4 compute; the rest are the other
    // stage arrangements (export, local, hull, geometry, domain), which aren't compiled early.
    // First uses still check it, and a type used otherwise stops being compiled early.
    precompile.type_stage.fill(UnguessedStage);
    precompile.type_stage[0] = HwStage::Fragment;
    precompile.type_stage[1] = HwStage::Vertex;
    precompile.type_stage[4] = HwStage::Compute;
    precompile.hold_ns = 15'000'000'000;
    if (const char* hold = std::getenv("SHADPS4_PRECOMPILE_HOLD_S")) {
        precompile.hold_ns = static_cast<u64>(std::strtod(hold, nullptr) * 1e9);
    }
    if (const char* order = std::getenv("SHADPS4_PRECOMPILE_ORDER")) {
        precompile.order_near = std::string_view{order} != "newest";
    }
    precompile.active = true;
    for (u32 i = 0; i < num_threads; ++i) {
        precompile.threads.emplace_back([this](std::stop_token stop) { RunPrecompile(stop); });
    }
    LOG_INFO(Render_Vulkan, "Compiling found shaders early on {} threads, {} first", num_threads,
             precompile.order_near ? "nearest the latest draws" : "newest found");
}

void PipelineCache::StopPrecompile() {
    if (!precompile.active) {
        return;
    }
    for (auto& thread : precompile.threads) {
        thread.request_stop();
    }
    precompile.cv.notify_all();
    precompile.threads.clear();
    const auto device = instance.GetDevice();
    for (auto& done : precompile.done) {
        for (const auto& m : done.program->modules) {
            if (shader_objects) {
                shader_objects->Forget(m.module);
            }
            device.destroyShaderModule(m.module);
        }
    }
    precompile.done.clear();
    precompile.active = false;
}

void PipelineCache::RunPrecompile(std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:Precompile");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    shader_scanner->WaitUntilLoaded();
    if (const auto saved = shader_scanner->SemanticBase()) {
        int unknown = -1;
        if (semantic_base.compare_exchange_strong(unknown, *saved)) {
            LOG_INFO(Render_Vulkan,
                     "Pixel shader inputs are their semantic less {}, as an "
                     "earlier run learned",
                     *saved);
        }
    }
    while (!stop.stop_requested()) {
        Precompile::Job job;
        {
            std::unique_lock lock{precompile.mutex};
            // The scan has no way to wake the workers, so they look for new finds now and then.
            const auto take_found = [&] {
                for (const u64 hash : shader_scanner->TakeFound(precompile.found_cursor)) {
                    if (!precompile.seen.insert(hash).second) {
                        continue;
                    }
                    // A stage saved by an earlier run needs no learning.
                    if (const auto saved = shader_scanner->StageOf(hash)) {
                        if (const auto stage = static_cast<HwStage>(*saved);
                            PrecompilesStage(stage)) {
                            precompile.queue.push_back(MakePrecompileJob(hash, stage));
                        }
                        continue;
                    }
                    const auto type = shader_scanner->TypeOf(hash).value_or(0) & 0xf;
                    const auto stage = precompile.type_stage[type];
                    if (precompile.type_conflict[type]) {
                        continue;
                    }
                    if (PrecompilesStage(stage)) {
                        precompile.queue.push_back(MakePrecompileJob(hash, stage));
                    }
                }
            };
            // Learning the semantic base can take long (until gameplay), so held shaders stop
            // waiting for it after a while and go with the fallback guess.
            const auto expire_hold = [&] {
                if (precompile.held.empty() || precompile.hold_expired.load() ||
                    VideoCore::PerfStats::NowNs() - precompile.first_held_ns < precompile.hold_ns) {
                    return;
                }
                precompile.hold_expired = true;
                LOG_INFO(Render_Vulkan,
                         "Pixel shader inputs not learned after {:.0f} s; compiling the {} held "
                         "pixel shaders with each one's lowest semantic as the base",
                         precompile.hold_ns / 1e9, precompile.held.size());
                RequeueHeldLocked();
            };
            take_found();
            expire_hold();
            while (precompile.queue.empty()) {
                precompile.cv.wait_for(lock, stop, std::chrono::milliseconds{250},
                                       [&] { return !precompile.queue.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                take_found();
                expire_hold();
            }
            const size_t index = PickPrecompileJob();
            job = precompile.queue[index];
            precompile.queue.erase(precompile.queue.begin() + index);
            if (precompile.used.contains(job.hash)) {
                // The game compiled it itself while it waited here.
                ++precompile.skipped_used;
                continue;
            }
            precompile.in_flight.insert(job.hash);
        }
        PrecompileOne(job);
        {
            std::scoped_lock lock{precompile.mutex};
            precompile.in_flight.erase(job.hash);
        }
        precompile.finished.notify_all();
    }
}

void PipelineCache::PrecompileOne(const Precompile::Job& job) {
    const u64 hash = job.hash;
    const HwStage hw_stage = job.stage;
    const u64 start = VideoCore::PerfStats::NowNs();
    const auto found = shader_scanner->Find(hash);
    const auto sw_stage = hw_stage == HwStage::Compute  ? SwStage::Compute
                          : hw_stage == HwStage::Vertex ? SwStage::Vertex
                                                        : SwStage::Fragment;
    if (!found) {
        return;
    }
    // Earlier runs' first uses say exactly what to compile.
    if (const auto states = shader_scanner->StatesOf(hash);
        !states.empty() && PrecompileRecorded(job, *found, states)) {
        return;
    }
    if (found->header.empty()) {
        return;
    }
    auto runtime_info = GuessRuntimeInfo(hw_stage, sw_stage, found->header);
    if (!runtime_info) {
        return;
    }
    if (hw_stage == HwStage::Fragment && runtime_info->hw.fs.num_inputs != 0 &&
        semantic_base.load(std::memory_order_relaxed) < 0 &&
        !precompile.hold_expired.load(std::memory_order_relaxed)) {
        // Its inputs are guessed best once draws have shown how semantics map to vertex
        // outputs, so it waits for that, for a while. Checked again under the lock that the
        // release takes, so none is left behind.
        std::unique_lock lock{precompile.mutex};
        if (semantic_base.load(std::memory_order_relaxed) < 0 &&
            !precompile.hold_expired.load(std::memory_order_relaxed)) {
            if (precompile.held.empty()) {
                precompile.first_held_ns = VideoCore::PerfStats::NowNs();
            }
            precompile.held.push_back(hash);
            return;
        }
        lock.unlock();
        runtime_info = GuessRuntimeInfo(hw_stage, sw_stage, found->header);
        if (!runtime_info) {
            return;
        }
    }
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    if (hw_stage == HwStage::Vertex) {
        fetch = GuessFetchShader(found->header);
        if (!fetch) {
            return;
        }
    }
    // Nothing is known of the draw's user data; the first use fills it in.
    const Shader::ShaderParams params{
        .user_data = NoUserData,
        .code = found->code,
        .hash = hash,
    };
    auto program = std::make_unique<Program>(hw_stage, sw_stage, params);
    auto& info = program->info;
    info.guess_state = true;
    info.guessed_fetch = fetch ? &*fetch : nullptr;
    thread_local Shader::Pools worker_pools;
    Shader::Backend::Bindings binding{};
    // The struct, the stage and even the code are guesses (the scan may catch bytes that aren't
    // a shader, or one half written), so a failed check in translation drops this shader
    // instead of stopping the emulator.
    std::vector<u32> spv;
    try {
        const Common::SoftAsserts soft_asserts;
        const auto ir_program =
            Shader::TranslateProgram(found->code, worker_pools, info, *runtime_info, profile);
        if (!GuessTranslated(info, hw_stage)) {
            throw Common::SoftAssertFailure{};
        }
        spv = Shader::Backend::SPIRV::EmitSPIRV(profile, *runtime_info, ir_program, binding);
    } catch (const Common::SoftAssertFailure&) {
        LOG_INFO(Render_Vulkan, "Not compiling {} shader {:#x} early: it doesn't translate as one",
                 hw_stage, hash);
        std::scoped_lock lock{precompile.mutex};
        ++precompile.rejected;
        return;
    }
    auto spec = Shader::StageSpecialization(info, *runtime_info, profile, {});
    info.guess_state = false;
    info.guessed_fetch = nullptr;

    Precompiled done{.hash = hash};
    const bool made_object = PrepareVariant(*program, done, info, std::move(spec), std::move(spv));
    program->prepared = true;
    done.program = std::move(program);

    std::scoped_lock lock{precompile.mutex};
    ++precompile.translated;
    precompile.shader_objects += made_object ? 1 : 0;
    precompile.compute_pipelines += static_cast<u32>(done.computes.size());
    precompile.worker_ns += VideoCore::PerfStats::NowNs() - start;
    precompile.done.push_back(std::move(done));
    precompile.has_done.store(true, std::memory_order_release);
}

bool PipelineCache::PrepareVariant(Program& program, Precompiled& done, const Shader::Info& info,
                                   Shader::StageSpecialization&& spec, std::vector<u32>&& spv) {
    const auto hw_stage = info.hw_stage;
    const auto device = instance.GetDevice();
    const auto module = CompileSPV(spv, device);
    Vulkan::SetObjectName(device, module, "{} (found early)",
                          GetShaderName(hw_stage, info.pgm_hash, program.modules.size()));
    const u64 perm_hash =
        HashCombine(info.pgm_hash, XXH3_64bits(spv.data(), spv.size() * sizeof(u32)));
    program.AddPermut(module, std::move(spec), perm_hash);

    // The driver's part: a shader object for a fragment shader, which every pipeline using it
    // shares; a whole pipeline for a compute shader, which is the shader alone.
    bool made_object = false;
    if ((hw_stage == HwStage::Fragment || hw_stage == HwStage::Vertex) && shader_objects) {
        shader_objects->Register(module, spv);
        StageHeapMappings mappings;
        mappings.Build(info, HeapGeometry::Of(instance));
        const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
            .requiredSubgroupSize = 64,
        };
        mappings.info.pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr;
        u64 driver_ns = 0;
        shader_objects->Get(module,
                            hw_stage == HwStage::Vertex ? vk::ShaderStageFlagBits::eVertex
                                                        : vk::ShaderStageFlagBits::eFragment,
                            &mappings.info, driver_ns);
        made_object = true;
    } else if (hw_stage == HwStage::Compute) {
        ComputePipeline::SerializationSupport sdata{};
        done.computes.emplace_back(perm_hash, std::make_unique<ComputePipeline>(
                                                  instance, scheduler, desc_heap, profile,
                                                  *pipeline_cache, ComputePipelineKey{perm_hash},
                                                  info, module, sdata, false, nullptr));
    } else if (shader_objects) {
        shader_objects->Register(module, spv);
    }
    program.prepared_spv.emplace_back(perm_hash, std::move(spv));
    return made_object;
}

namespace {

/// Recorded draw states: the version, then the stages, the runtime info, the user data, the
/// flattened user data, the fetch shader and the vertex buffer V#s. Laid out in this build's
/// structs, so the version changes with their sizes.
constexpr u32 DrawStateVersion = (1u << 24) ^ u32(sizeof(Shader::RuntimeInfo)) ^
                                 u32(sizeof(Shader::Gcn::VertexAttribute) << 16) ^
                                 u32(sizeof(AmdGpu::Buffer) << 20);
static_assert(std::is_trivially_copyable_v<Shader::RuntimeInfo>);
static_assert(std::is_trivially_copyable_v<Shader::Gcn::VertexAttribute>);
static_assert(std::is_trivially_copyable_v<AmdGpu::Buffer>);

struct DrawState {
    HwStage hw_stage;
    SwStage sw_stage;
    Shader::RuntimeInfo runtime_info;
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data;
    std::vector<u32> flat_buf;
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::vector<std::pair<u32, AmdGpu::Buffer>> vertex_buffers;
};

template <typename T>
void PutState(std::vector<u8>& out, const T& value) {
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

std::optional<DrawState> ParseDrawState(std::span<const u8> in) {
    u32 version{};
    u8 hw{};
    u8 sw{};
    DrawState state{};
    u32 flat_size{};
    if (!GetState(in, version) || version != DrawStateVersion || !GetState(in, hw) ||
        !GetState(in, sw) || !GetState(in, state.runtime_info) || !GetState(in, state.user_data) ||
        !GetState(in, flat_size) || in.size() < flat_size * sizeof(u32)) {
        return std::nullopt;
    }
    state.hw_stage = static_cast<HwStage>(hw);
    state.sw_stage = static_cast<SwStage>(sw);
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
            Shader::Gcn::VertexAttribute attribute{};
            if (!GetState(in, attribute)) {
                return std::nullopt;
            }
            fetch.attributes.push_back(attribute);
        }
    }
    u32 num_buffers{};
    if (!GetState(in, num_buffers) || num_buffers > Shader::IR::NumParams) {
        return std::nullopt;
    }
    for (u32 i = 0; i < num_buffers; ++i) {
        std::pair<u32, AmdGpu::Buffer> buffer{};
        if (!GetState(in, buffer.first) || !GetState(in, buffer.second)) {
            return std::nullopt;
        }
        state.vertex_buffers.push_back(buffer);
    }
    return state;
}

} // namespace

void PipelineCache::RecordDrawState(const Shader::Info& info,
                                    const Shader::RuntimeInfo& runtime_info,
                                    const Shader::Gcn::FetchShaderData& fetch) {
    if (!precompile.active || !PrecompilesStage(PrecompileStageOf(info.hw_stage, info.sw_stage))) {
        return;
    }
    std::vector<u8> out;
    PutState(out, DrawStateVersion);
    PutState(out, static_cast<u8>(info.hw_stage));
    PutState(out, static_cast<u8>(info.sw_stage));
    PutState(out, runtime_info);
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::copy_n(info.user_data.begin(), std::min(info.user_data.size(), user_data.size()),
                user_data.begin());
    PutState(out, user_data);
    PutState(out, static_cast<u32>(info.flattened_ud_buf.size()));
    const auto* flat = reinterpret_cast<const u8*>(info.flattened_ud_buf.data());
    out.insert(out.end(), flat, flat + info.flattened_ud_buf.size() * sizeof(u32));
    PutState(out, fetch.size);
    if (fetch.size != 0) {
        PutState(out, fetch.vertex_offset_sgpr);
        PutState(out, fetch.instance_offset_sgpr);
        PutState(out, static_cast<u32>(fetch.attributes.size()));
        for (const auto& attribute : fetch.attributes) {
            PutState(out, attribute);
        }
    }
    PutState(out, static_cast<u32>(fetch.attributes.size()));
    for (const auto& attribute : fetch.attributes) {
        PutState(out, static_cast<u32>(attribute.dword_offset));
        PutState(out, info.ReadUdReg<AmdGpu::Buffer>(attribute.sgpr_base, attribute.dword_offset));
    }
    shader_scanner->NoteState(info.pgm_hash, std::move(out));
}

bool PipelineCache::PrecompileRecorded(const Precompile::Job& job,
                                       const VideoCore::ShaderScanner::Shader& found,
                                       const std::vector<std::vector<u8>>& states) {
    const u64 start = VideoCore::PerfStats::NowNs();
    thread_local Shader::Pools replay_pools;
    std::unique_ptr<Program> program;
    Precompiled done{.hash = job.hash};
    u32 objects = 0;
    u32 variants = 0;
    for (const auto& blob : states) {
        const auto state = ParseDrawState(blob);
        if (!state || state->hw_stage != job.stage) {
            continue;
        }
        const Shader::ShaderParams params{
            .user_data = state->user_data,
            .code = found.code,
            .hash = job.hash,
        };
        Shader::Info info{state->hw_stage, state->sw_stage, params};
        const Shader::Info::ReplayState replay{
            .flat_buf = state->flat_buf,
            .fetch = state->fetch ? &*state->fetch : nullptr,
            .vertex_buffers = state->vertex_buffers,
        };
        info.replay = &replay;
        auto runtime_info = state->runtime_info;
        Shader::Backend::Bindings binding{};
        std::vector<u32> spv;
        try {
            const Common::SoftAsserts soft_asserts;
            const auto ir_program =
                Shader::TranslateProgram(found.code, replay_pools, info, runtime_info, profile);
            spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
        } catch (const Common::SoftAssertFailure&) {
            LOG_INFO(Render_Vulkan, "Recorded draw state of {} shader {:#x} doesn't translate",
                     job.stage, job.hash);
            continue;
        }
        auto spec = Shader::StageSpecialization(info, runtime_info, profile, {});
        info.replay = nullptr;
        info.user_data = NoUserData;
        if (program && std::ranges::find(program->modules, spec, &Program::Module::spec) !=
                           program->modules.end()) {
            continue;
        }
        const bool first = !program;
        if (first) {
            program = std::make_unique<Program>(state->hw_stage, state->sw_stage, params);
        }
        objects += PrepareVariant(*program, done, info, std::move(spec), std::move(spv)) ? 1 : 0;
        if (first) {
            program->info = std::move(info);
        }
        ++variants;
    }
    if (!program) {
        return false;
    }
    program->info.user_data = NoUserData;
    program->prepared = true;
    done.program = std::move(program);
    std::scoped_lock lock{precompile.mutex};
    ++precompile.translated;
    ++precompile.replayed;
    precompile.replayed_variants += variants;
    precompile.shader_objects += objects;
    precompile.compute_pipelines += static_cast<u32>(done.computes.size());
    precompile.worker_ns += VideoCore::PerfStats::NowNs() - start;
    precompile.done.push_back(std::move(done));
    precompile.has_done.store(true, std::memory_order_release);
    return true;
}

void PipelineCache::ReleaseHeldPrecompiles() {
    if (!precompile.active) {
        return;
    }
    {
        std::scoped_lock lock{precompile.mutex};
        LOG_INFO(Render_Vulkan, "{} pixel shaders held for their inputs are queued again",
                 precompile.held.size());
        RequeueHeldLocked();
    }
    precompile.cv.notify_all();
}

void PipelineCache::RequeueHeldLocked() {
    for (const u64 hash : precompile.held) {
        if (!precompile.used.contains(hash)) {
            precompile.queue.push_back(MakePrecompileJob(hash, HwStage::Fragment));
        }
    }
    precompile.held.clear();
}

std::vector<u64> PipelineCache::SavedTilingKeys() {
    shader_scanner->WaitUntilLoaded();
    return shader_scanner->TilingKeys();
}

void PipelineCache::NoteTilingKey(u64 key) {
    shader_scanner->NoteTilingKey(key);
}

void PipelineCache::JoinPrecompile(u64 hash) {
    // A worker already part way through the program is done sooner than starting again here.
    // Bounded, in case it's stuck on a slow driver compile.
    constexpr auto MaxWait = std::chrono::milliseconds{100};
    if (program_cache.contains(hash)) {
        return;
    }
    std::unique_lock lock{precompile.mutex};
    if (!precompile.in_flight.contains(hash)) {
        return;
    }
    const u64 start = VideoCore::PerfStats::NowNs();
    precompile.finished.wait_for(lock, MaxWait,
                                 [&] { return !precompile.in_flight.contains(hash); });
    const u64 waited = VideoCore::PerfStats::NowNs() - start;
    ++precompile.joined;
    precompile.joined_ns += waited;
    VideoCore::PerfStats::frames.AddShaderTime(waited);
}

void PipelineCache::TakePrecompiled() {
    if (!precompile.has_done.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<Precompiled> done;
    {
        std::scoped_lock lock{precompile.mutex};
        done.swap(precompile.done);
        precompile.has_done.store(false, std::memory_order_relaxed);
    }
    const auto device = instance.GetDevice();
    for (auto& item : done) {
        const auto [it, fresh] = program_cache.try_emplace(item.hash);
        if (!fresh) {
            // The game got there first.
            for (const auto& m : item.program->modules) {
                if (shader_objects) {
                    shader_objects->Forget(m.module);
                }
                device.destroyShaderModule(m.module);
            }
            ++precompile.duplicate;
            continue;
        }
        it.value() = std::move(item.program);
        for (auto& [perm_hash, compute] : item.computes) {
            compute_pipelines.try_emplace(ComputePipelineKey{perm_hash}, std::move(compute));
        }
    }
}

PipelineCache::Precompile::Job PipelineCache::MakePrecompileJob(u64 hash, HwStage stage) {
    const auto place = shader_scanner->PlaceOf(hash);
    return {hash, stage, place.addr, place.use_order};
}

size_t PipelineCache::PickPrecompileJob() {
    const size_t newest = precompile.queue.size() - 1;
    if (newest == 0) {
        return 0;
    }
    // What earlier runs drew is what this one will draw first.
    size_t earliest = newest;
    u32 earliest_order = std::numeric_limits<u32>::max();
    for (size_t i = 0; i <= newest; ++i) {
        const u32 order = precompile.queue[i].use_order;
        if (order != 0 && order < earliest_order) {
            earliest_order = order;
            earliest = i;
        }
    }
    if (earliest_order != std::numeric_limits<u32>::max()) {
        return earliest;
    }
    if (!precompile.order_near) {
        return newest;
    }
    const auto recent = shader_scanner->RecentUses();
    if (std::ranges::all_of(recent, [](u64 addr) { return addr == 0; })) {
        return newest;
    }
    // Nearest to any of the latest first uses; ties and unknown addresses go to the newest.
    size_t best = newest;
    u64 best_distance = std::numeric_limits<u64>::max();
    for (size_t i = precompile.queue.size(); i-- > 0;) {
        const u64 addr = precompile.queue[i].addr;
        if (addr == 0) {
            continue;
        }
        for (const u64 use : recent) {
            if (use == 0) {
                continue;
            }
            const u64 distance = addr > use ? addr - use : use - addr;
            if (distance < best_distance) {
                best_distance = distance;
                best = i;
            }
        }
    }
    return best;
}

void PipelineCache::NotePrecompileType(u64 hash, HwStage hw_stage) {
    const auto type = shader_scanner->TypeOf(hash);
    std::scoped_lock lock{precompile.mutex};
    precompile.seen.insert(hash);
    precompile.used.insert(hash);
    if (!type) {
        return;
    }
    const u8 t = *type & 0xf;
    const auto name = [](HwStage s) {
        return s == UnguessedStage ? std::string{"other-stage (not compiled early)"}
                                   : fmt::format("{}", s);
    };
    auto& stage = precompile.type_stage[t];
    if (precompile.type_conflict[t]) {
        return;
    }
    if (stage != hw_stage) {
        precompile.type_conflict[t] = true;
        LOG_WARNING(Render_Vulkan,
                    "Shader binary type {} is used as {}, not {}; not compiling it early", t,
                    name(hw_stage), name(stage));
    }
}

void PipelineCache::NotePreparedUse(Program& program, u64 hash, bool served,
                                    const Shader::StageSpecialization& spec, u64 perm_hash) {
    program.prepared = false;
    if (shader_scanner) {
        shader_scanner->NoteFirstUse(
            hash, static_cast<u8>(
                      PrecompileStageOf(spec.runtime_info.hw_stage, spec.runtime_info.sw_stage)));
    }
    if (served) {
        ++precompile.served;
        const auto module_it =
            std::ranges::find(program.modules, perm_hash, &Program::Module::hash);
        RegisterShaderMeta(program.info, spec.fetch_shader_data, spec, perm_hash,
                           std::distance(program.modules.begin(), module_it));
        if (const auto spv = std::ranges::find(program.prepared_spv, perm_hash,
                                               &std::pair<u64, std::vector<u32>>::first);
            spv != program.prepared_spv.end()) {
            RegisterShaderBinary(std::move(spv->second), perm_hash);
        }
        LOG_INFO(Render_Vulkan, "{} shader {:#x} was compiled early and served its first use",
                 program.info.hw_stage, hash);
    } else {
        ++precompile.missed;
        LOG_INFO(Render_Vulkan,
                 "{} shader {:#x} was compiled early, but its first use differs in: {}",
                 program.info.hw_stage, hash,
                 fmt::join(SpecDifferences(spec, program.modules[0].spec), ", "));
    }
    program.prepared_spv = {};
}

/// A pixel shader's input semantics from its shader struct: after the stage registers, a count
/// byte (padded to a word), the input usage slots, then a u16 per input (semantic in the low
/// byte, flat shading in bit 10).
static std::optional<std::vector<u16>> PsInputSemantics(std::span<const u8> header) {
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
std::optional<Shader::Gcn::FetchShaderData> PipelineCache::GuessFetchShader(
    std::span<const u8> header) {
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

void PipelineCache::LearnSemanticBase(u64 hash, const Shader::RuntimeInfo& runtime_info) {
    const auto found = shader_scanner->Find(hash);
    if (!found || found->header.empty()) {
        return;
    }
    const auto semantics = PsInputSemantics(found->header);
    const auto& fs = runtime_info.hw.fs;
    if (!semantics || semantics->size() != fs.num_inputs || fs.num_inputs == 0) {
        return;
    }
    // How the fallback guess, each shader's lowest semantic as the base, would have done.
    u32 lowest = 0xff;
    for (u32 i = 0; i < fs.num_inputs; ++i) {
        lowest = std::min<u32>(lowest, (*semantics)[i] & 0xff);
    }
    bool lowest_right = true;
    for (u32 i = 0; i < fs.num_inputs; ++i) {
        if (!fs.inputs[i].is_default &&
            ((*semantics)[i] & 0xff) - lowest != fs.inputs[i].param_index) {
            lowest_right = false;
        }
    }
    ++(lowest_right ? semantic_min_right : semantic_min_wrong);
    // The vertex stage's outputs seem to be ordered by semantic, so an input's parameter is
    // its semantic less the first one the game exports. Learned from what draws use.
    for (u32 i = 0; i < fs.num_inputs; ++i) {
        if (fs.inputs[i].is_default) {
            continue;
        }
        const int base = int((*semantics)[i] & 0xff) - int(fs.inputs[i].param_index);
        if (const int known = semantic_base.load(std::memory_order_relaxed); known >= 0) {
            if (base != known) {
                ++semantic_base_misses;
            }
            continue;
        }
        if (base == semantic_base_candidate) {
            if (++semantic_base_votes == 3) {
                semantic_base.store(base, std::memory_order_relaxed);
                shader_scanner->SetSemanticBase(base);
                ReleaseHeldPrecompiles();
                LOG_INFO(Render_Vulkan, "Pixel shader inputs are their semantic less {}", base);
            }
        } else {
            semantic_base_candidate = base;
            semantic_base_votes = 1;
        }
    }
}

std::optional<Shader::RuntimeInfo> PipelineCache::GuessRuntimeInfo(
    HwStage hw_stage, SwStage sw_stage, std::span<const u8> header) const {
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
        int base = semantic_base.load(std::memory_order_relaxed);
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
                fs.color_buffers[i] = profile.fewer_shader_variants
                                          ? ShaderColorBuffer(Shader::PsColorBuffer{})
                                          : Shader::PsColorBuffer{};
                continue;
            }
            const auto target = UnboundTarget(export_format);
            fs.color_buffers[i] =
                profile.fewer_shader_variants ? ShaderColorBuffer(target) : target;
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
        if (profile.fewer_shader_variants && instance.IsVertexInputDynamicState()) {
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

void PipelineCache::CheckGuess(HwStage hw_stage, SwStage sw_stage,
                               const Shader::ShaderParams& params,
                               Shader::RuntimeInfo& runtime_info,
                               const Shader::Backend::Bindings& start,
                               const Shader::StageSpecialization& spec, u64 perm_hash) {
    guess_stats.changed = true;
    if (sw_stage != SwStage::Fragment && sw_stage != SwStage::Compute) {
        ++guess_stats.skipped;
        // Vertex shaders need their fetch shader, which the game makes from the vertex shader's
        // struct at run time. Logs a few structs next to the fetch shader the draw used, to see
        // whether it can be made from the struct alone.
        const auto found = shader_scanner->Find(params.hash);
        if (sw_stage == SwStage::Vertex && found && !found->header.empty() &&
            guess_stats.dumped_vertex < 30) {
            ++guess_stats.dumped_vertex;
            std::string bytes;
            for (const u8 b : found->header) {
                bytes += fmt::format("{:02x}", b);
            }
            std::string attributes;
            for (const auto& a : spec.fetch_shader_data.attributes) {
                attributes +=
                    fmt::format(" [v{} n{} s{} +{} inst{} off{} fmt{}/{}]", a.dest_vgpr,
                                a.num_elements, a.sgpr_base, a.dword_offset, a.instance_data,
                                a.inst_offset, a.data_format, a.num_format);
            }
            LOG_INFO(Render_Vulkan,
                     "Shader struct of {} shader {:#x} ({} bytes): {}; fetch shader: vertex "
                     "offset s{}, instance offset s{},{}",
                     hw_stage, params.hash, found->header.size(), bytes,
                     spec.fetch_shader_data.vertex_offset_sgpr,
                     spec.fetch_shader_data.instance_offset_sgpr, attributes);
        }
        if (sw_stage == SwStage::Vertex && hw_stage == HwStage::Vertex && found &&
            !found->header.empty()) {
            CheckVertexGuess(params, found->header, start, spec, perm_hash);
        }
        return;
    }
    ++guess_stats.checked;
    const auto found = shader_scanner->Find(params.hash);
    if (found->code.size() != params.code.size() || !std::ranges::equal(found->code, params.code)) {
        LOG_INFO(Render_Vulkan, "Shader scan's copy of {} shader {:#x} differs: {} vs {} words",
                 hw_stage, params.hash, found->code.size(), params.code.size());
        ++guess_stats.code_differs;
    }
    // Translates with nothing known of the resources (no user data), and returns the SPIR-V's
    // hash and how the specialisation differs from the draw's.
    const auto translate = [&](Shader::RuntimeInfo& guess_runtime) {
        Shader::Info guess(hw_stage, sw_stage, params);
        guess.guess_state = true;
        auto binding = start;
        const auto ir_program =
            Shader::TranslateProgram(params.code, pools, guess, guess_runtime, profile);
        const auto spv =
            Shader::Backend::SPIRV::EmitSPIRV(profile, guess_runtime, ir_program, binding);
        const u64 hash =
            HashCombine(params.hash, XXH3_64bits(spv.data(), spv.size() * sizeof(u32)));
        const auto guess_spec = Shader::StageSpecialization(guess, guess_runtime, profile, start);
        return std::make_pair(hash, SpecDifferences(spec, guess_spec));
    };
    // Counted per kind, not per binding: "buf3.format" counts as "buf.format".
    const auto count_kinds = [](const std::vector<std::string>& parts, auto& counts) {
        for (const auto& part : parts) {
            std::string kind;
            std::ranges::copy_if(part, std::back_inserter(kind), [](char c) {
                return !std::isdigit(static_cast<unsigned char>(c));
            });
            ++counts[kind];
        }
    };

    // First with the draw's register state, so this part measures guessing resources only.
    const auto [hash, wrong] = translate(runtime_info);
    if (wrong.empty()) {
        ++guess_stats.spec_matched;
    }
    count_kinds(wrong, guess_stats.wrong);
    auto& guessed = guessed_hashes[params.hash];
    if (hash == perm_hash) {
        ++guess_stats.matched;
    } else {
        guessed.resources = hash;
    }

    // Then with the register state guessed from the shader struct as well: nothing from the
    // draw, which is what a translation ahead of the first draw has.
    std::optional<Shader::RuntimeInfo> header_runtime;
    if (!found->header.empty()) {
        header_runtime = GuessRuntimeInfo(hw_stage, sw_stage, found->header);
    }
    if (!header_runtime) {
        ++guess_stats.no_registers;
        if (guessed.resources == 0) {
            guessed_hashes.erase(params.hash);
        }
        LOG_INFO(Render_Vulkan, "Guess for {} shader {:#x}: {} with draw registers{}", hw_stage,
                 params.hash, hash == perm_hash ? "matches" : "differs",
                 found->header.empty() ? ", no shader struct" : ", shader struct too short");
        return;
    }
    const auto [full_hash, full_wrong] = translate(*header_runtime);
    // Only the parts that come from registers, to see which of them the struct gets wrong.
    std::vector<std::string> register_wrong;
    std::ranges::copy_if(full_wrong, std::back_inserter(register_wrong), [&](const auto& part) {
        return std::ranges::find(wrong, part) == wrong.end();
    });
    count_kinds(register_wrong, guess_stats.registers_wrong);
    if (full_hash == perm_hash) {
        ++guess_stats.full_matched;
    } else {
        guessed.full = full_hash;
    }
    if (guessed.resources == 0 && guessed.full == 0) {
        guessed_hashes.erase(params.hash);
    }
    LOG_INFO(Render_Vulkan,
             "Guess for {} shader {:#x}: {} with draw registers, {} with struct registers; "
             "resources differ in: {}; registers differ in: {}",
             hw_stage, params.hash, hash == perm_hash ? "matches" : "differs",
             full_hash == perm_hash ? "matches" : "differs",
             wrong.empty() ? std::string{"nothing"} : fmt::format("{}", fmt::join(wrong, ", ")),
             register_wrong.empty() ? std::string{"nothing"}
                                    : fmt::format("{}", fmt::join(register_wrong, ", ")));
    // A few raw structs next to what the draw had, to work out the parts still guessed.
    if (!register_wrong.empty() && guess_stats.dumped < 40) {
        ++guess_stats.dumped;
        std::string bytes;
        for (const u8 b : found->header) {
            bytes += fmt::format("{:02x}", b);
        }
        std::string inputs;
        if (hw_stage == HwStage::Fragment) {
            const auto& fs = runtime_info.hw.fs;
            for (u32 i = 0; i < fs.num_inputs; ++i) {
                const auto& in = fs.inputs[i];
                inputs += fmt::format(" {}{}{}", in.param_index, in.is_flat ? "f" : "",
                                      in.is_default ? fmt::format("d{}", in.default_value) : "");
            }
        }
        LOG_INFO(Render_Vulkan, "Shader struct of {} shader {:#x} ({} bytes): {}; draw inputs:{}",
                 hw_stage, params.hash, found->header.size(), bytes, inputs);
    }
}

void PipelineCache::CheckVertexGuess(const Shader::ShaderParams& params, std::span<const u8> header,
                                     const Shader::Backend::Bindings& start,
                                     const Shader::StageSpecialization& spec, u64 perm_hash) {
    ++guess_stats.vertex_checked;
    auto runtime = GuessRuntimeInfo(HwStage::Vertex, SwStage::Vertex, header);
    auto fetch = GuessFetchShader(header);
    if (!runtime || !fetch) {
        ++guess_stats.vertex_unguessable;
        return;
    }
    if (*fetch == spec.fetch_shader_data) {
        ++guess_stats.vertex_fetch_matched;
    }
    Shader::Info guess(HwStage::Vertex, SwStage::Vertex, params);
    guess.guess_state = true;
    guess.guessed_fetch = &*fetch;
    auto binding = start;
    const auto ir_program = Shader::TranslateProgram(params.code, pools, guess, *runtime, profile);
    const auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, *runtime, ir_program, binding);
    const u64 hash = HashCombine(params.hash, XXH3_64bits(spv.data(), spv.size() * sizeof(u32)));
    const auto guess_spec = Shader::StageSpecialization(guess, *runtime, profile, start);
    const auto wrong = SpecDifferences(spec, guess_spec);
    for (const auto& part : wrong) {
        std::string kind;
        std::ranges::copy_if(part, std::back_inserter(kind),
                             [](char c) { return !std::isdigit(static_cast<unsigned char>(c)); });
        ++guess_stats.vertex_wrong[kind];
    }
    const bool matched = hash == perm_hash;
    guess_stats.vertex_matched += matched ? 1 : 0;
    LOG_INFO(Render_Vulkan, "Guess for vs shader {:#x} with its struct: {}; differs in: {}",
             params.hash, matched ? "matches" : "differs",
             wrong.empty() ? std::string{"nothing"} : fmt::format("{}", fmt::join(wrong, ", ")));
}

void PipelineCache::LogGuessStats() {
    if (!guess_stats.changed) {
        return;
    }
    guess_stats.changed = false;
    const auto list = [](const std::map<std::string, u32>& counts) {
        std::vector<std::string> parts;
        for (const auto& [kind, count] : counts) {
            parts.push_back(fmt::format("{} {}", kind, count));
        }
        return parts.empty() ? std::string{"none"} : fmt::format("{}", fmt::join(parts, ", "));
    };
    LOG_INFO(Render_Vulkan,
             "Guessed translation so far: {} programs checked, {} skipped, {} scan copies "
             "differed. With draw registers: {} same SPIR-V, {} same specialisation, {} matched "
             "a later variant; wrong: {}. With struct registers: {} same SPIR-V, {} matched a "
             "later variant, {} had no struct; registers wrong: {}",
             guess_stats.checked, guess_stats.skipped, guess_stats.code_differs,
             guess_stats.matched, guess_stats.spec_matched, guess_stats.matched_later,
             list(guess_stats.wrong), guess_stats.full_matched, guess_stats.full_matched_later,
             guess_stats.no_registers, list(guess_stats.registers_wrong));
    LOG_INFO(Render_Vulkan,
             "Guessed vertex shaders so far: {} checked, {} same SPIR-V, {} same fetch shader, {} "
             "without a usable struct; wrong: {}",
             guess_stats.vertex_checked, guess_stats.vertex_matched,
             guess_stats.vertex_fetch_matched, guess_stats.vertex_unguessable,
             list(guess_stats.vertex_wrong));
    LOG_INFO(Render_Vulkan,
             "Pixel shader input semantics: base {}, {} inputs off it; each shader's lowest "
             "semantic as the base would be right for {} draws and wrong for {}",
             semantic_base.load(std::memory_order_relaxed), semantic_base_misses,
             semantic_min_right, semantic_min_wrong);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    // Background creations may still use the module destroyed below.
    compiler->WaitIdle();
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                if (shader_objects) {
                    shader_objects->Forget(m.module);
                }
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                if (shader_objects) {
                    shader_objects->Register(m.module, spv_code);
                }
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
