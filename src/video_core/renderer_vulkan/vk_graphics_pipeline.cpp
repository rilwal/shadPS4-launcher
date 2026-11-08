// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/runtime_buffer_format.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_compiler.h"
#include "video_core/renderer_vulkan/vk_resource_heaps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_objects.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::Backend::SPIRV::AuxShaderType;

static constexpr std::array LogicalStageToStageBit = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

static Shader::SwStage StageBitToLogicalStage(vk::ShaderStageFlagBits bit) {
    const auto it = std::ranges::find(LogicalStageToStageBit, bit);
    ASSERT(it != LogicalStageToStageBit.end());
    return static_cast<Shader::SwStage>(std::distance(LogicalStageToStageBit.begin(), it));
}

/// Everything vkCreateGraphicsPipelines reads, kept in one place so that a worker can create the
/// pipeline after the constructor has returned. The layout still belongs to the pipeline, so a
/// probe that runs after the pipeline is ready relies on pipelines only being destroyed with the
/// compiler idle (ReplaceShader waits, and the compiler stops first at shutdown).
struct GraphicsPipeline::Build {
    vk::Device device;
    vk::PipelineCache cache;
    std::string name;
    VertexInputs<vk::VertexInputAttributeDescription> vertex_attributes;
    VertexInputs<vk::VertexInputBindingDescription> vertex_bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    vk::PipelineVertexInputDivisorStateCreateInfo divisor_state;
    vk::PipelineVertexInputStateCreateInfo vertex_input_info;
    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    vk::PipelineTessellationStateCreateInfo tessellation_state;
    vk::StructureChain<vk::PipelineRasterizationStateCreateInfo,
                       vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT,
                       vk::PipelineRasterizationDepthClipStateCreateInfoEXT>
        raster_chain;
    vk::PipelineMultisampleStateCreateInfo multisampling;
    vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control;
    vk::PipelineViewportStateCreateInfo viewport_info;
    boost::container::static_vector<vk::DynamicState, 32> dynamic_states;
    vk::PipelineDynamicStateCreateInfo dynamic_info;
    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
        shader_stages;
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci;
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats;
    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    vk::AttachmentSampleCountInfoAMD mixed_samples;
    vk::PipelineRenderingCreateInfo pipeline_rendering_ci;
    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> attachments;
    vk::PipelineColorBlendStateCreateInfo color_blending;
    vk::PipelineDepthStencilStateCreateInfo depth_stencil_info;
    vk::GraphicsPipelineCreateInfo pipeline_info;
    /// With the descriptor heap: where each stage finds its descriptors, and the heap flag.
    bool uses_heap{};
    std::array<StageHeapMappings, MaxShaderStages> heap_mappings;
    vk::PipelineCreateFlags2CreateInfo flags2{};
    /// With shader objects: the code of each helper stage we generate, by shader stage index,
    /// copied so that a worker can create it after the constructor.
    std::array<std::vector<u32>, MaxShaderStages> generated;
};

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    const Shader::Gcn::FetchShaderData* fetch_shader_, std::span<const vk::ShaderModule> modules,
    SerializationSupport& sdata, bool preloading, PipelineCompileOptions options)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache}, key{key_} {
    if (fetch_shader_) {
        fetch_shader = *fetch_shader_;
    }

    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    const auto debug_str = GetDebugString();
    auto build = std::make_unique<Build>();
    Build& b = *build;
    b.device = device;
    b.cache = pipeline_cache;
    b.name = debug_str;
    // With the descriptor heap there are no layouts: each stage maps its own bindings.
    b.uses_heap = instance.UsesDescriptorHeap();
    // Shader objects need the heap, so they never need a layout either.
    ShaderObjects* const shader_objects = options.shader_objects;
    ASSERT(!shader_objects || b.uses_heap);

    if (!b.uses_heap) {
        BuildDescSetLayout(preloading);
        const vk::PushConstantRange push_constants = {
            .stageFlags = AllGraphicsStageBits,
            .offset = 0,
            .size = sizeof(Shader::PushData),
        };

        const vk::DescriptorSetLayout set_layout = *desc_layout;
        const vk::PipelineLayoutCreateInfo layout_info = {
            .setLayoutCount = 1U,
            .pSetLayouts = &set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constants,
        };
        auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
        ASSERT_MSG(layout_result == vk::Result::eSuccess,
                   "Failed to create graphics pipeline layout: {}", vk::to_string(layout_result));
        pipeline_layout = std::move(layout);
        SetObjectName(device, *pipeline_layout, "Graphics PipelineLayout {}", debug_str);
    }

    if (!preloading) {
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        if (!instance.IsVertexInputDynamicState()) {
            const auto& vs_info = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
            GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                            guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1);
        }
    }

    b.vertex_attributes = sdata.vertex_attributes;
    b.vertex_bindings = sdata.vertex_bindings;
    b.divisors = sdata.divisors;
    b.divisor_state = vk::PipelineVertexInputDivisorStateCreateInfo{
        .vertexBindingDivisorCount = static_cast<u32>(b.divisors.size()),
        .pVertexBindingDivisors = b.divisors.data(),
    };

    b.vertex_input_info = vk::PipelineVertexInputStateCreateInfo{
        .pNext = b.divisors.empty() ? nullptr : &b.divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(b.vertex_bindings.size()),
        .pVertexBindingDescriptions = b.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(b.vertex_attributes.size()),
        .pVertexAttributeDescriptions = b.vertex_attributes.data(),
    };

    const auto topology = LiverpoolToVK::PrimitiveType(key.prim_type);
    b.input_assembly = vk::PipelineInputAssemblyStateCreateInfo{
        .topology = topology,
    };

    const bool is_rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == AmdGpu::PrimitiveType::QuadList;
    b.tessellation_state = vk::PipelineTessellationStateCreateInfo{
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    // Filled in place, as the chain links its own members.
    auto& raster = b.raster_chain.get<vk::PipelineRasterizationStateCreateInfo>();
    raster.depthClampEnable =
        key.depth_clamp_enable && (!key.depth_clip_enable || instance.IsDepthClipEnableSupported());
    raster.rasterizerDiscardEnable = false;
    raster.polygonMode = LiverpoolToVK::PolygonMode(key.polygon_mode);
    raster.lineWidth = 1.0f;
    b.raster_chain.get<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>()
        .provokingVertexMode = key.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First
                                   ? vk::ProvokingVertexModeEXT::eFirstVertex
                                   : vk::ProvokingVertexModeEXT::eLastVertex;
    b.raster_chain.get<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>().depthClipEnable =
        key.depth_clip_enable;

    if (!instance.IsProvokingVertexSupported()) {
        b.raster_chain.unlink<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>();
    }
    if (!instance.IsDepthClipEnableSupported()) {
        b.raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    if (!preloading) {
        const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
        sdata.multisampling = {
            .rasterizationSamples = LiverpoolToVK::NumSamples(
                key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
            .sampleShadingEnable =
                fs_info.addr_flags.persp_sample_ena || fs_info.addr_flags.linear_sample_ena,
        };
    }
    b.multisampling = sdata.multisampling;

    b.clip_control = vk::PipelineViewportDepthClipControlCreateInfoEXT{
        .negativeOneToOne = key.clip_space == AmdGpu::ClipSpace::MinusWToW,
    };

    b.viewport_info = vk::PipelineViewportStateCreateInfo{
        .pNext = instance.IsDepthClipControlSupported() ? &b.clip_control : nullptr,
    };

    auto& dynamic_states = b.dynamic_states;
    dynamic_states = {
        vk::DynamicState::eViewportWithCount,  vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eBlendConstants,     vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,  vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilOp,          vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,          vk::DynamicState::eRasterizerDiscardEnable,
        vk::DynamicState::eLineWidth,          vk::DynamicState::ePrimitiveRestartEnable,
    };

    if (instance.IsDepthBoundsSupported()) {
        dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
        dynamic_states.push_back(vk::DynamicState::eDepthBounds);
    }
    if (instance.IsDynamicColorWriteMaskSupported()) {
        dynamic_states.push_back(vk::DynamicState::eColorWriteMaskEXT);
    }
    if (instance.IsVertexInputDynamicState()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputEXT);
    } else if (!sdata.vertex_bindings.empty()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputBindingStride);
    }

    b.dynamic_info = vk::PipelineDynamicStateCreateInfo{
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    auto& shader_stages = b.shader_stages;
    auto stage = u32(Shader::SwStage::Vertex);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Geometry);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationControl);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto type = is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS;
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(type, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module =
                shader_objects ? vk::ShaderModule{} : CompileSPV(sdata.tcs, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationEval);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                AuxShaderType::PassthroughTES, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module =
                shader_objects ? vk::ShaderModule{} : CompileSPV(sdata.tes, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Fragment);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs.clip_distance_emulation) {
        if (!preloading) {
            const auto& vs = runtime_infos[static_cast<u32>(Shader::SwStage::Vertex)].hw.vs;

            sdata.fragment = Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs.outputs);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shader_objects ? vk::ShaderModule{}
                                     : CompileSPV(sdata.fragment, instance.GetDevice()),
            .pName = "main",
        });
    }

    b.subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .requiredSubgroupSize = 64,
    };
    for (auto& stage : shader_stages) {
        stage.pNext = instance.IsSubgroupSize64Supported() ? &b.subgroup_size_ci : nullptr;
    }
    if (b.uses_heap) {
        const auto geometry = HeapGeometry::Of(instance);
        for (size_t i = 0; i < shader_stages.size(); ++i) {
            auto& stage_ci = shader_stages[i];
            const auto sw_stage = StageBitToLogicalStage(stage_ci.stage);
            const auto* stage_info = infos[u32(sw_stage)];
            if (!stage_info) {
                // Our own helper stages read no descriptors.
                continue;
            }
            auto& mappings = b.heap_mappings[i];
            mappings.Build(*stage_info, geometry);
            mappings.info.pNext = stage_ci.pNext;
            stage_ci.pNext = &mappings.info;
        }
    }

    const auto depth_format =
        instance.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    auto& color_formats = b.color_formats;
    for (s32 i = 0; i < key.num_color_attachments; ++i) {
        const auto& col_buf = key.color_buffers[i];
        const auto format = LiverpoolToVK::SurfaceFormat(col_buf.data_format, col_buf.num_format);
        const auto color_format =
            instance.GetSupportedFormat(format, vk::FormatFeatureFlagBits2::eColorAttachment);
        if (!instance.IsFormatSupported(color_format,
                                        vk::FormatFeatureFlagBits2::eColorAttachment)) {
            LOG_WARNING(Render_Vulkan,
                        "color buffer format {} does not support COLOR_ATTACHMENT_BIT",
                        vk::to_string(color_format));
        }
        color_formats[i] = color_format;
    }

    auto& color_samples = b.color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 num_samples) {
        return num_samples ? LiverpoolToVK::NumSamples(num_samples, instance.GetColorSampleCounts())
                           : vk::SampleCountFlagBits::e1;
    });
    b.mixed_samples = vk::AttachmentSampleCountInfoAMD{
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };

    b.pipeline_rendering_ci = vk::PipelineRenderingCreateInfo{
        .pNext = instance.IsMixedDepthSamplesSupported() ? &b.mixed_samples : nullptr,
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid
                                     ? depth_format
                                     : vk::Format::eUndefined,
        .stencilAttachmentFormat = key.stencil_format != AmdGpu::DepthBuffer::StencilFormat::Invalid
                                       ? depth_format
                                       : vk::Format::eUndefined,
    };

    auto& attachments = b.attachments;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        const auto& control = key.blend_controls[i];

        const auto src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
        const auto color_blend = LiverpoolToVK::BlendOp(control.color_func);

        const auto src_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                                   : src_color;
        const auto dst_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                                   : dst_color;
        const auto alpha_blend =
            control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : color_blend;

        attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = control.enable,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = color_blend,
            .srcAlphaBlendFactor = src_alpha,
            .dstAlphaBlendFactor = dst_alpha,
            .alphaBlendOp = alpha_blend,
            .colorWriteMask =
                instance.IsDynamicColorWriteMaskSupported()
                    ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                    : key.write_masks[i],
        };

        // The shader squares its color output for this attachment (see PsColorBuffer), so the
        // factors must not scale the operands again.
        if (key.color_buffers[i].blend_self_scale) {
            LOG_WARNING(
                Render_Vulkan,
                "Emulating scaled min/max blend with squared shader output on attachment {}", i);
            attachments[i].srcColorBlendFactor = vk::BlendFactor::eOne;
            attachments[i].dstColorBlendFactor = vk::BlendFactor::eOne;
        }

        // On GCN GPU there is an additional mask which allows to control color components exported
        // from a pixel shader. A situation possible, when the game may mask out the alpha channel,
        // while it is still need to be used in blending ops. For such cases, HW will default alpha
        // to 1 and perform the blending, while shader normally outputs 0 in the last component.
        // Unfortunatelly, Vulkan doesn't provide any control on blend inputs, so below we detecting
        // such cases and override alpha value in order to emulate HW behaviour.
        const auto has_alpha_masked_out =
            (key.cb_shader_mask.GetMask(i) & AmdGpu::ColorBufferMask::ComponentA) == 0;
        const auto has_src_alpha_in_src_blend = src_color == vk::BlendFactor::eSrcAlpha ||
                                                src_color == vk::BlendFactor::eOneMinusSrcAlpha;
        const auto has_src_alpha_in_dst_blend = dst_color == vk::BlendFactor::eSrcAlpha ||
                                                dst_color == vk::BlendFactor::eOneMinusSrcAlpha;
        if (has_alpha_masked_out && has_src_alpha_in_src_blend) {
            attachments[i].srcColorBlendFactor = src_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }
        if (has_alpha_masked_out && has_src_alpha_in_dst_blend) {
            attachments[i].dstColorBlendFactor = dst_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }

        auto& attachment = attachments[i];
        LiverpoolToVK::NormalizeMinMaxBlend(attachment.colorBlendOp, attachment.srcColorBlendFactor,
                                            attachment.dstColorBlendFactor,
                                            key.color_buffers[i].num_format);
        LiverpoolToVK::NormalizeMinMaxBlend(attachment.alphaBlendOp, attachment.srcAlphaBlendFactor,
                                            attachment.dstAlphaBlendFactor,
                                            key.color_buffers[i].num_format);

        const auto color_scaled_min_max =
            (attachment.colorBlendOp == vk::BlendOp::eMin ||
             attachment.colorBlendOp == vk::BlendOp::eMax) &&
            (attachment.srcColorBlendFactor != vk::BlendFactor::eOne ||
             attachment.dstColorBlendFactor != vk::BlendFactor::eOne);
        const auto alpha_scaled_min_max =
            (attachment.alphaBlendOp == vk::BlendOp::eMin ||
             attachment.alphaBlendOp == vk::BlendOp::eMax) &&
            (attachment.srcAlphaBlendFactor != vk::BlendFactor::eOne ||
             attachment.dstAlphaBlendFactor != vk::BlendFactor::eOne);
        if (control.enable && (color_scaled_min_max || alpha_scaled_min_max)) {
            LOG_WARNING(
                Render_Vulkan,
                "Unimplemented use of min/max blend op with blend factor not equal to one.");
        }
    }

    b.color_blending = vk::PipelineColorBlendStateCreateInfo{
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != AmdGpu::ColorControl::LogicOp::Copy,
        .logicOp = LiverpoolToVK::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    if (shader_objects) {
        // Helper stages are made here, so their code is the shader object's key.
        for (size_t i = 0; i < shader_stages.size(); ++i) {
            const auto sw_stage = StageBitToLogicalStage(shader_stages[i].stage);
            if (infos[u32(sw_stage)]) {
                continue;
            }
            switch (sw_stage) {
            case Shader::SwStage::TessellationControl:
                b.generated[i] = sdata.tcs;
                break;
            case Shader::SwStage::TessellationEval:
                b.generated[i] = sdata.tes;
                break;
            case Shader::SwStage::Fragment:
                b.generated[i] = sdata.fragment;
                break;
            default:
                UNREACHABLE();
            }
        }
        if (!options.compiler) {
            CreateShaderObjects(b, *shader_objects);
            if (options.on_created) {
                options.on_created(driver_ns);
            }
            return;
        }
        // Workers create the shaders that don't exist yet; draws skip or wait until then.
        state.store(State::Pending, std::memory_order_relaxed);
        compile_ticket = options.compiler->Submit(
            [this, build = std::move(build), shader_objects, compiler = options.compiler,
             on_created = std::move(options.on_created)](bool drop) {
                if (drop) {
                    Finish(State::Dropped);
                    return;
                }
                CreateShaderObjects(*build, *shader_objects);
                const u64 created_ns = driver_ns;
                // The pipeline may be destroyed from here on, so nothing below touches it.
                Finish(State::Ready);
                compiler->Notify();
                if (on_created) {
                    on_created(created_ns);
                }
            });
        return;
    }

    // Required by spec unless VK_EXT_extended_dynamic_state3 is supported.
    // In practice, we use dynamic state for all of it.
    b.flags2 = vk::PipelineCreateFlags2CreateInfo{
        .pNext = &b.pipeline_rendering_ci,
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };
    b.pipeline_info = vk::GraphicsPipelineCreateInfo{
        .pNext = b.uses_heap ? static_cast<const void*>(&b.flags2) : &b.pipeline_rendering_ci,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &b.vertex_input_info : nullptr,
        .pInputAssemblyState = &b.input_assembly,
        .pTessellationState = &b.tessellation_state,
        .pViewportState = &b.viewport_info,
        .pRasterizationState = &b.raster_chain.get(),
        .pMultisampleState = &b.multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &b.depth_stencil_info : nullptr,
        .pColorBlendState = &b.color_blending,
        .pDynamicState = &b.dynamic_info,
        .layout = b.uses_heap ? vk::PipelineLayout{} : *pipeline_layout,
    };

    if (options.compiler && options.tiered) {
        Create(b, false);
        if (options.on_created) {
            options.on_created(driver_ns);
        }
        upgrading.store(true, std::memory_order_relaxed);
        options.compiler->Submit([this, build = std::move(build),
                                  on_upgraded = std::move(options.on_upgraded)](bool drop) {
            u64 optimized_ns = 0;
            if (!drop) {
                const u64 start = VideoCore::PerfStats::NowNs();
                auto [result, pipe] =
                    build->device.createGraphicsPipelineUnique(build->cache, build->pipeline_info);
                optimized_ns = VideoCore::PerfStats::NowNs() - start;
                if (result == vk::Result::eSuccess) {
                    SetObjectName(build->device, *pipe, "Graphics Pipeline {}", build->name);
                    Upgrade(std::move(pipe));
                }
            }
            // The pipeline may be destroyed from here on, so nothing below touches it.
            upgrading.store(false, std::memory_order_release);
            upgrading.notify_all();
            if (!drop && on_upgraded) {
                on_upgraded(optimized_ns);
            }
        });
        return;
    }

    if (!options.compiler) {
        Create(b);
        if (options.on_created) {
            options.on_created(driver_ns);
        }
        if (options.prober) {
            options.prober->Submit([build = std::move(build), optimized_ns = driver_ns,
                                    on_probed = std::move(options.on_probed)](bool drop) {
                if (!drop) {
                    Probe(*build, optimized_ns, on_probed);
                }
            });
        }
        return;
    }

    state.store(State::Pending, std::memory_order_relaxed);
    compile_ticket = options.compiler->Submit(
        [this, build = std::move(build), compiler = options.compiler, prober = options.prober,
         on_created = std::move(options.on_created),
         on_probed = std::move(options.on_probed)](bool drop) {
            if (drop) {
                Finish(State::Dropped);
                return;
            }
            Create(*build);
            const u64 optimized_ns = driver_ns;
            // The pipeline may be destroyed from here on, so nothing below touches it.
            Finish(State::Ready);
            compiler->Notify();
            if (on_created) {
                on_created(optimized_ns);
            }
            if (prober) {
                Probe(*build, optimized_ns, on_probed);
            }
        });
}

void GraphicsPipeline::CreateShaderObjects(const Build& b, ShaderObjects& objects) {
    auto st = std::make_unique<ShaderObjectState>();
    // Every stage the device has is bound, null when the draw doesn't use it.
    st->stages[st->num_stages++] = vk::ShaderStageFlagBits::eVertex;
    if (instance.IsTessellationSupported()) {
        st->stages[st->num_stages++] = vk::ShaderStageFlagBits::eTessellationControl;
        st->stages[st->num_stages++] = vk::ShaderStageFlagBits::eTessellationEvaluation;
    }
    if (instance.IsGeometryStageSupported()) {
        st->stages[st->num_stages++] = vk::ShaderStageFlagBits::eGeometry;
    }
    st->stages[st->num_stages++] = vk::ShaderStageFlagBits::eFragment;

    u64 created_ns = 0;
    for (size_t i = 0; i < b.shader_stages.size(); ++i) {
        const auto& stage_ci = b.shader_stages[i];
        const auto slot = std::ranges::find(st->stages, stage_ci.stage) - st->stages.begin();
        ASSERT(slot < st->num_stages);
        st->shaders[slot] =
            !b.generated[i].empty()
                ? objects.GetGenerated(b.generated[i], stage_ci.stage, stage_ci.pNext, created_ns)
                : objects.Get(stage_ci.module, stage_ci.stage, stage_ci.pNext, created_ns);
        st->tessellation |= stage_ci.stage == vk::ShaderStageFlagBits::eTessellationEvaluation;
    }
    driver_ns = created_ns;

    const auto& raster = b.raster_chain.get<vk::PipelineRasterizationStateCreateInfo>();
    st->topology = st->tessellation ? vk::PrimitiveTopology::ePatchList : b.input_assembly.topology;
    st->patch_control_points = b.tessellation_state.patchControlPoints;
    st->samples = b.multisampling.rasterizationSamples;
    st->polygon_mode = raster.polygonMode;
    st->depth_clamp = raster.depthClampEnable;
    st->depth_clip =
        b.raster_chain.get<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>().depthClipEnable;
    st->provoking_vertex =
        b.raster_chain.get<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>()
            .provokingVertexMode;
    st->negative_one_to_one = b.clip_control.negativeOneToOne;
    st->logic_op_enable = b.color_blending.logicOpEnable;
    st->logic_op = b.color_blending.logicOp;
    st->num_attachments = b.color_blending.attachmentCount;
    for (u32 i = 0; i < st->num_attachments; ++i) {
        const auto& a = b.attachments[i];
        st->blend_enables[i] = a.blendEnable;
        st->blend_equations[i] = vk::ColorBlendEquationEXT{
            .srcColorBlendFactor = a.srcColorBlendFactor,
            .dstColorBlendFactor = a.dstColorBlendFactor,
            .colorBlendOp = a.colorBlendOp,
            .srcAlphaBlendFactor = a.srcAlphaBlendFactor,
            .dstAlphaBlendFactor = a.dstAlphaBlendFactor,
            .alphaBlendOp = a.alphaBlendOp,
        };
    }
    shader_state = std::move(st);
}

void GraphicsPipeline::BindShaderObjects(vk::CommandBuffer cmdbuf,
                                         std::optional<ShaderObjectState>& bound) const {
    const auto& st = *shader_state;
    const ShaderObjectState* prev = bound ? &*bound : nullptr;
    if (!prev) {
        cmdbuf.bindShadersEXT(st.num_stages, st.stages.data(), st.shaders.data());
    } else {
        // Every pipeline binds the same stage slots, so only the shaders that differ change.
        std::array<vk::ShaderStageFlagBits, 5> stages;
        std::array<vk::ShaderEXT, 5> shaders;
        u32 count = 0;
        for (u32 i = 0; i < st.num_stages; ++i) {
            if (st.shaders[i] != prev->shaders[i]) {
                stages[count] = st.stages[i];
                shaders[count++] = st.shaders[i];
            }
        }
        if (count != 0) {
            cmdbuf.bindShadersEXT(count, stages.data(), shaders.data());
        }
    }
    if (!prev || prev->topology != st.topology) {
        cmdbuf.setPrimitiveTopology(st.topology);
    }
    if (st.tessellation &&
        (!prev || !prev->tessellation || prev->patch_control_points != st.patch_control_points)) {
        cmdbuf.setPatchControlPointsEXT(st.patch_control_points);
        // What a pipeline without a domain origin state uses.
        cmdbuf.setTessellationDomainOriginEXT(vk::TessellationDomainOrigin::eUpperLeft);
    }
    if (!prev || prev->samples != st.samples) {
        cmdbuf.setRasterizationSamplesEXT(st.samples);
        const std::array<vk::SampleMask, 2> sample_mask = {~0u, ~0u};
        cmdbuf.setSampleMaskEXT(st.samples, sample_mask.data());
    }
    if (!prev) {
        cmdbuf.setAlphaToCoverageEnableEXT(false);
    }
    if (!prev || prev->polygon_mode != st.polygon_mode) {
        cmdbuf.setPolygonModeEXT(st.polygon_mode);
    }
    if (instance.IsDepthClampSupported() && (!prev || prev->depth_clamp != st.depth_clamp)) {
        cmdbuf.setDepthClampEnableEXT(st.depth_clamp);
    }
    if (instance.IsDepthClipEnableSupported() && (!prev || prev->depth_clip != st.depth_clip)) {
        cmdbuf.setDepthClipEnableEXT(st.depth_clip);
    }
    if (instance.IsProvokingVertexSupported() &&
        (!prev || prev->provoking_vertex != st.provoking_vertex)) {
        cmdbuf.setProvokingVertexModeEXT(st.provoking_vertex);
    }
    if (instance.IsDepthClipControlSupported() &&
        (!prev || prev->negative_one_to_one != st.negative_one_to_one)) {
        cmdbuf.setDepthClipNegativeOneToOneEXT(st.negative_one_to_one);
    }
    if (instance.IsLogicOpSupported()) {
        if (!prev || prev->logic_op_enable != st.logic_op_enable) {
            cmdbuf.setLogicOpEnableEXT(st.logic_op_enable);
        }
        if (!prev || prev->logic_op != st.logic_op) {
            cmdbuf.setLogicOpEXT(st.logic_op);
        }
    }
    if (instance.IsNvFramebufferMixedSamplesSupported() && !prev) {
        cmdbuf.setCoverageModulationModeNV(vk::CoverageModulationModeNV::eNone);
        cmdbuf.setCoverageModulationTableEnableNV(false);
    }
    if (st.num_attachments != 0 &&
        (!prev || prev->num_attachments < st.num_attachments ||
         !std::equal(st.blend_enables.begin(), st.blend_enables.begin() + st.num_attachments,
                     prev->blend_enables.begin()) ||
         !std::equal(st.blend_equations.begin(), st.blend_equations.begin() + st.num_attachments,
                     prev->blend_equations.begin()))) {
        cmdbuf.setColorBlendEnableEXT(0, st.num_attachments, st.blend_enables.data());
        cmdbuf.setColorBlendEquationEXT(0, st.num_attachments, st.blend_equations.data());
    }
    bound = st;
}

GraphicsPipeline::~GraphicsPipeline() {
    WaitIdle();
}

void GraphicsPipeline::Create(const Build& b, bool optimize) {
    auto info = b.pipeline_info;
    auto flags2 = b.flags2;
    if (!optimize) {
        // With flags2 chained, the driver reads the flags from there instead.
        info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
        flags2.flags |= vk::PipelineCreateFlagBits2::eDisableOptimization;
        if (b.uses_heap) {
            info.pNext = &flags2;
        }
    }
    const u64 start = VideoCore::PerfStats::NowNs();
    auto [pipeline_result, pipe] = b.device.createGraphicsPipelineUnique(b.cache, info);
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create graphics pipeline: {}",
               vk::to_string(pipeline_result));
    driver_ns = VideoCore::PerfStats::NowNs() - start;
    pipeline = std::move(pipe);
    SetObjectName(b.device, *pipeline, "Graphics Pipeline {}", b.name);
}

void GraphicsPipeline::Probe(const Build& b, u64 optimized_ns,
                             const std::function<void(u64, u64)>& on_probed) {
    auto info = b.pipeline_info;
    auto flags2 = b.flags2;
    info.flags |= vk::PipelineCreateFlagBits::eDisableOptimization;
    flags2.flags |= vk::PipelineCreateFlagBits2::eDisableOptimization;
    if (b.uses_heap) {
        info.pNext = &flags2;
    }
    const u64 start = VideoCore::PerfStats::NowNs();
    auto [result, pipe] = b.device.createGraphicsPipelineUnique({}, info);
    const u64 unoptimized_ns = VideoCore::PerfStats::NowNs() - start;
    if (result == vk::Result::eSuccess && on_probed) {
        on_probed(optimized_ns, unoptimized_ns);
    }
}

vk::Format GraphicsPipeline::VertexInputFormat(const AmdGpu::Buffer& buffer) const {
    if (!Shader::UsesRuntimeVertexFormat(profile.runtime_vertex_formats, buffer)) {
        return LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
    }
    // The element's raw bits, which the shader converts by the format in its stage table.
    switch (Shader::RuntimeFormatElementSize(*Shader::DescribeRuntimeBufferFormat(buffer))) {
    case 1:
        return vk::Format::eR8Uint;
    case 2:
        return vk::Format::eR16Uint;
    case 4:
        return vk::Format::eR32Uint;
    case 8:
        return vk::Format::eR32G32Uint;
    case 12:
        return vk::Format::eR32G32B32Uint;
    case 16:
        return vk::Format::eR32G32B32A32Uint;
    default:
        UNREACHABLE();
    }
}

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (fetch_shader.Empty() || fetch_shader.attributes.empty()) {
        return;
    }
    const auto& vs_info = GetStage(Shader::SwStage::Vertex);
    for (u32 semantic = 0; semantic < fetch_shader.attributes.size(); ++semantic) {
        const auto& attrib = fetch_shader.attributes[semantic];
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs_info);
        attributes.push_back(Attribute{
            .location = semantic,
            .binding = semantic,
            .format = VertexInputFormat(buffer),
            .offset = 0,
        });
        bindings.push_back(Binding{
            .binding = semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
        });
        const u32 divisor = step_rate == InstanceIdType::OverStepRate0
                                ? step_rate_0
                                : (step_rate == InstanceIdType::OverStepRate1 ? step_rate_1 : 1);
        if constexpr (std::is_same_v<Binding, vk::VertexInputBindingDescription2EXT>) {
            bindings.back().divisor = divisor;
        } else if (step_rate != InstanceIdType::None) {
            divisors.push_back(vk::VertexInputBindingDivisorDescriptionEXT{
                .binding = semantic,
                .divisor = divisor,
            });
        }
        guest_buffers.emplace_back(buffer);
    }
}

// Declare templated GetVertexInputs for necessary types.
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>& attributes,
    VertexInputs<vk::VertexInputBindingDescription>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;

void GraphicsPipeline::BuildDescSetLayout(bool preloading) {
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    u32 binding{};

    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto stage_bit = LogicalStageToStageBit[u32(stage->sw_stage)];
        for (const auto& buffer : stage->buffers) {
            const auto sharp =
                preloading ? AmdGpu::Buffer{}
                           : buffer.GetSharp(*stage); // See for the comment in compute PL creation
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        for (const auto& image : stage->images) {
            const u32 num_bindings = image.NumBindings(*stage);
            bindings.push_back({
                .binding = binding,
                .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                                   : vk::DescriptorType::eSampledImage,
                .descriptorCount = num_bindings,
                .stageFlags = stage_bit,
            });
            binding += num_bindings;
        }
        for (const auto& sampler : stage->samplers) {
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
    }
    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [layout_result, layout] =
        instance.GetDevice().createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics descriptor set layout: {}", vk::to_string(layout_result));
    desc_layout = std::move(layout);
}

} // namespace Vulkan
