// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/info.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_resource_heaps.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading /*=false*/,
                                 PipelineCompiler* compiler /*=nullptr*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::SwStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();

    // Everything vkCreateComputePipelines reads, kept together so that a worker can create the
    // pipeline after the constructor has returned.
    struct Build {
        vk::Device device;
        vk::PipelineCache cache;
        std::string name;
        vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci;
        StageHeapMappings heap_mappings;
        vk::PipelineCreateFlags2CreateInfo flags2;
        vk::ComputePipelineCreateInfo ci;
    };
    auto build = std::make_unique<Build>();
    build->subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &build->subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    const auto device = instance.GetDevice();
    // With the descriptor heap there are no layouts: the shader maps its own bindings.
    const bool uses_heap = instance.UsesDescriptorHeap();
    auto& heap_mappings = build->heap_mappings;
    vk::PipelineShaderStageCreateInfo stage_ci = shader_ci;
    if (uses_heap) {
        heap_mappings.Build(*info, HeapGeometry::Of(instance));
        heap_mappings.info.pNext = stage_ci.pNext;
        stage_ci.pNext = &heap_mappings.info;
    } else {
        u32 binding{};
        boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
        for (const auto& buffer : info->buffers) {
            // During deserialization, we don't have access to the UD to fetch sharp data. To
            // address this properly we need to track shaprs or portion of them in `sdata`, but
            // since we're interested only in "is storage" flag (which is not even effective atm),
            // we can take a shortcut there.
            const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            });
        }
        for (const auto& image : info->images) {
            const u32 num_bindings = image.NumBindings(*info);
            bindings.push_back({
                .binding = binding,
                .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                                   : vk::DescriptorType::eSampledImage,
                .descriptorCount = num_bindings,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            });
            binding += num_bindings;
        }
        for (const auto& sampler : info->samplers) {
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            });
        }

        const vk::PushConstantRange push_constants = {
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(Shader::PushData),
        };

        uses_push_descriptors = binding < instance.MaxPushDescriptors();
        const auto flags = uses_push_descriptors
                               ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                               : vk::DescriptorSetLayoutCreateFlagBits{};
        const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
            .flags = flags,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        };
        auto [descriptor_set_result, descriptor_set] =
            device.createDescriptorSetLayoutUnique(desc_layout_ci);
        ASSERT_MSG(descriptor_set_result == vk::Result::eSuccess,
                   "Failed to create compute descriptor set layout: {}",
                   vk::to_string(descriptor_set_result));
        desc_layout = std::move(descriptor_set);

        const vk::DescriptorSetLayout set_layout = *desc_layout;
        const vk::PipelineLayoutCreateInfo layout_info = {
            .setLayoutCount = 1U,
            .pSetLayouts = &set_layout,
            .pushConstantRangeCount = 1U,
            .pPushConstantRanges = &push_constants,
        };
        auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
        ASSERT_MSG(layout_result == vk::Result::eSuccess,
                   "Failed to create compute pipeline layout: {}", vk::to_string(layout_result));
        pipeline_layout = std::move(layout);
        SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);
    }

    build->device = device;
    build->cache = pipeline_cache;
    build->name = debug_str;
    build->flags2 = vk::PipelineCreateFlags2CreateInfo{
        .flags = vk::PipelineCreateFlagBits2::eDescriptorHeapEXT,
    };
    build->ci = vk::ComputePipelineCreateInfo{
        .pNext = uses_heap ? &build->flags2 : nullptr,
        .stage = stage_ci,
        .layout = uses_heap ? vk::PipelineLayout{} : *pipeline_layout,
    };
    const auto create = [this](const Build& b) {
        const u64 start = VideoCore::PerfStats::NowNs();
        auto [pipeline_result, pipe] = b.device.createComputePipelineUnique(b.cache, b.ci);
        ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
                   vk::to_string(pipeline_result));
        driver_ns = VideoCore::PerfStats::NowNs() - start;
        pipeline = std::move(pipe);
        SetObjectName(b.device, *pipeline, "Compute Pipeline {}", b.name);
    };
    if (!compiler) {
        create(*build);
        return;
    }
    // Created ahead of its dispatch on the workers; the dispatch waits for it if it must.
    state.store(State::Pending, std::memory_order_relaxed);
    compile_ticket =
        compiler->Submit([this, build = std::move(build), create, compiler](bool drop) {
            if (drop) {
                Finish(State::Dropped);
                return;
            }
            create(*build);
            // The pipeline may be destroyed from here on, so nothing below touches it.
            Finish(State::Ready);
            compiler->Notify();
        });
}

ComputePipeline::~ComputePipeline() {
    WaitIdle();
}

} // namespace Vulkan
