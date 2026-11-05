// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <ranges>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/scope_exit.h"
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
            info.hw.fs.color_buffers[i] = graphics_key.color_buffers[i];
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
    };
    // A quarter of the cores: the game's own threads and the GPU thread need the rest.
    const u32 compile_threads = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
    compiler = std::make_unique<PipelineCompiler>(compile_threads);
    if (instance.UsesShaderObjects()) {
        shader_objects = std::make_unique<ShaderObjects>(instance);
    }
    LOG_INFO(Render_Vulkan, "{} pipeline compile threads", compile_threads);
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() = default;

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
                 "Frames over {:.1f} s: {} ({:.1f} FPS), {} over 33 ms, worst {:.1f} ms. "
                 "Compiled {} graphics + {} compute pipelines from {} shaders ({} permutations, "
                 "{} only for binding offsets): recompile "
                 "{:.1f} ms, driver {:.1f} ms, worst pipeline {:.1f} ms. "
                 "Async {}: {} sent to workers, {} created there (driver {:.1f} ms, worst "
                 "{:.1f} ms, ready at worst {:.1f} ms after the draw), {} draws skipped, "
                 "{} draws waited for a new target ({} ran on the GPU thread), waited {:.1f} ms, "
                 "{} pending",
                 seconds, frames.count, frames.count / seconds, frames.hitches,
                 frames.worst_ns / 1e6, stats.graphics, stats.compute, stats.shaders,
                 stats.permutations, stats.binding_permutations, stats.recompile_ns / 1e6,
                 stats.driver_ns / 1e6, stats.worst_ns / 1e6, AsyncModeName(GetAsyncMode()),
                 stats.background, bg_created, bg_driver_ns / 1e6, bg_worst_driver_ns / 1e6,
                 bg_worst_latency_ns / 1e6, stats.skipped_draws, stats.waited_draws, stats.ran_now,
                 stats.waited_ns / 1e6, compiler->Pending());
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

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
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
        ++compile_stats.shaders;
    };

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    if (shader_objects) {
        shader_objects->Register(module, is_patched ? std::span<const u32>{*patch} : spv);
    }
    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(hw_stage, sw_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
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
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
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
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    return std::make_tuple(&program->info, module, perm_hash);
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
