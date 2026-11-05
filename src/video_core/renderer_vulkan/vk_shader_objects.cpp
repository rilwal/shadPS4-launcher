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
    for (auto& [_, slot] : by_module) {
        if (slot->shader) {
            device.destroyShaderEXT(slot->shader);
        }
    }
    for (auto& [_, slot] : generated) {
        if (slot->shader) {
            device.destroyShaderEXT(slot->shader);
        }
    }
}

void ShaderObjects::Register(vk::ShaderModule module, std::span<const u32> code) {
    auto slot = std::make_shared<Slot>();
    slot->code.assign(code.begin(), code.end());
    std::scoped_lock lock{mutex};
    by_module[static_cast<VkShaderModule>(module)] = std::move(slot);
}

void ShaderObjects::Forget(vk::ShaderModule module) {
    std::scoped_lock lock{mutex};
    const auto it = by_module.find(static_cast<VkShaderModule>(module));
    if (it == by_module.end()) {
        return;
    }
    if (it->second->shader) {
        instance.GetDevice().destroyShaderEXT(it->second->shader);
    }
    by_module.erase(it);
}

vk::ShaderEXT ShaderObjects::Get(vk::ShaderModule module, vk::ShaderStageFlagBits stage,
                                 const void* next, u64& driver_ns) {
    std::shared_ptr<Slot> slot;
    {
        std::scoped_lock lock{mutex};
        const auto it = by_module.find(static_cast<VkShaderModule>(module));
        ASSERT_MSG(it != by_module.end(), "No SPIR-V kept for a {} shader module",
                   vk::to_string(stage));
        slot = it->second;
    }
    return Resolve(*slot, stage, next, driver_ns);
}

vk::ShaderEXT ShaderObjects::GetGenerated(std::span<const u32> code, vk::ShaderStageFlagBits stage,
                                          const void* next, u64& driver_ns) {
    const u64 key = XXH3_64bits(code.data(), code.size_bytes()) ^ static_cast<u64>(stage);
    std::shared_ptr<Slot> slot;
    {
        std::scoped_lock lock{mutex};
        auto& entry = generated[key];
        if (!entry) {
            entry = std::make_shared<Slot>();
            entry->code.assign(code.begin(), code.end());
        }
        slot = entry;
    }
    return Resolve(*slot, stage, next, driver_ns);
}

vk::ShaderEXT ShaderObjects::Resolve(Slot& slot, vk::ShaderStageFlagBits stage, const void* next,
                                     u64& driver_ns) {
    bool made = false;
    std::call_once(slot.once, [&] {
        slot.shader = Create(slot.code, stage, next, driver_ns);
        made = true;
    });
    if (!made) {
        reused.fetch_add(1, std::memory_order_relaxed);
    }
    return slot.shader;
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
    created.fetch_add(1, std::memory_order_relaxed);
    return shader;
}

} // namespace Vulkan
