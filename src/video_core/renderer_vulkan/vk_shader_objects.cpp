// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>

#include "common/assert.h"
#include "video_core/perf_stats.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_objects.h"

namespace Vulkan {

ShaderObjects::ShaderObjects(const Instance& instance_) : instance{instance_} {}

ShaderObjects::~ShaderObjects() {
    const auto device = instance.GetDevice();
    for (auto& [_, entry] : by_module) {
        if (entry.shader) {
            device.destroyShaderEXT(entry.shader);
        }
    }
    for (auto& [_, shader] : generated) {
        device.destroyShaderEXT(shader);
    }
}

void ShaderObjects::Register(vk::ShaderModule module, std::span<const u32> code) {
    auto& entry = by_module[static_cast<VkShaderModule>(module)];
    entry.code.assign(code.begin(), code.end());
}

void ShaderObjects::Forget(vk::ShaderModule module) {
    const auto it = by_module.find(static_cast<VkShaderModule>(module));
    if (it == by_module.end()) {
        return;
    }
    if (it->second.shader) {
        instance.GetDevice().destroyShaderEXT(it->second.shader);
    }
    by_module.erase(it);
}

vk::ShaderEXT ShaderObjects::Get(vk::ShaderModule module, vk::ShaderStageFlagBits stage,
                                 const void* next, u64& driver_ns) {
    const auto it = by_module.find(static_cast<VkShaderModule>(module));
    ASSERT_MSG(it != by_module.end(), "No SPIR-V kept for a {} shader module",
               vk::to_string(stage));
    auto& entry = it.value();
    if (entry.shader) {
        ++counts.reused;
        return entry.shader;
    }
    entry.shader = Create(entry.code, stage, next, driver_ns);
    return entry.shader;
}

vk::ShaderEXT ShaderObjects::GetGenerated(std::span<const u32> code, vk::ShaderStageFlagBits stage,
                                          const void* next, u64& driver_ns) {
    const u64 key = XXH3_64bits(code.data(), code.size_bytes()) ^ static_cast<u64>(stage);
    const auto [it, is_new] = generated.try_emplace(key);
    if (!is_new) {
        ++counts.reused;
        return it->second;
    }
    it.value() = Create(code, stage, next, driver_ns);
    return it->second;
}

vk::ShaderEXT ShaderObjects::Create(std::span<const u32> code, vk::ShaderStageFlagBits stage,
                                    const void* next, u64& driver_ns) {
    // Shaders aren't linked, so each one is shared by every draw whatever the other stages are.
    vk::ShaderStageFlags next_stages{};
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex:
        next_stages = vk::ShaderStageFlagBits::eFragment;
        if (instance.IsTessellationSupported()) {
            next_stages |= vk::ShaderStageFlagBits::eTessellationControl;
        }
        if (instance.IsGeometryStageSupported()) {
            next_stages |= vk::ShaderStageFlagBits::eGeometry;
        }
        break;
    case vk::ShaderStageFlagBits::eTessellationControl:
        next_stages = vk::ShaderStageFlagBits::eTessellationEvaluation;
        break;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        next_stages = vk::ShaderStageFlagBits::eFragment;
        if (instance.IsGeometryStageSupported()) {
            next_stages |= vk::ShaderStageFlagBits::eGeometry;
        }
        break;
    case vk::ShaderStageFlagBits::eGeometry:
        next_stages = vk::ShaderStageFlagBits::eFragment;
        break;
    default:
        break;
    }
    const vk::ShaderCreateInfoEXT info = {
        .pNext = next,
        .flags = vk::ShaderCreateFlagBitsEXT::eDescriptorHeap,
        .stage = stage,
        .nextStage = next_stages,
        .codeType = vk::ShaderCodeTypeEXT::eSpirv,
        .codeSize = code.size_bytes(),
        .pCode = code.data(),
        .pName = "main",
    };
    const u64 start = VideoCore::PerfStats::NowNs();
    auto [result, shader] = instance.GetDevice().createShaderEXT(info);
    driver_ns += VideoCore::PerfStats::NowNs() - start;
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to create {} shader object: {}",
               vk::to_string(stage), vk::to_string(result));
    ++counts.created;
    return shader;
}

} // namespace Vulkan
