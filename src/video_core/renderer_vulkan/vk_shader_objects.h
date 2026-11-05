// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <utility>
#include <vector>
#include <tsl/robin_map.h>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

/// Shaders compiled once through VK_EXT_shader_object, so that every draw that uses a shader
/// shares it instead of compiling it again inside each pipeline. Only the GPU thread uses it.
class ShaderObjects {
public:
    explicit ShaderObjects(const Instance& instance);
    ~ShaderObjects();

    ShaderObjects(const ShaderObjects&) = delete;
    ShaderObjects& operator=(const ShaderObjects&) = delete;

    /// Keeps the SPIR-V behind a module, which the shader object is made from later.
    void Register(vk::ShaderModule module, std::span<const u32> code);

    /// Drops the module's shader object and code, for a module that's being replaced.
    void Forget(vk::ShaderModule module);

    /// The shader object for a guest shader's module, created the first time. `next` is the
    /// stage's create info chain (descriptor mappings, subgroup size). Adds the driver's time.
    vk::ShaderEXT Get(vk::ShaderModule module, vk::ShaderStageFlagBits stage, const void* next,
                      u64& driver_ns);

    /// The shader object for a helper stage that a pipeline generates, keyed by its code.
    vk::ShaderEXT GetGenerated(std::span<const u32> code, vk::ShaderStageFlagBits stage,
                               const void* next, u64& driver_ns);

    /// Shader objects created and reused since the last call, for the compile summary.
    struct Counts {
        u32 created{};
        u32 reused{};
    };
    Counts TakeCounts() noexcept {
        return std::exchange(counts, {});
    }

private:
    vk::ShaderEXT Create(std::span<const u32> code, vk::ShaderStageFlagBits stage, const void* next,
                         u64& driver_ns);

    const Instance& instance;
    struct Entry {
        std::vector<u32> code;
        vk::ShaderEXT shader;
    };
    tsl::robin_map<VkShaderModule, Entry> by_module;
    tsl::robin_map<u64, vk::ShaderEXT> generated;
    Counts counts;
};

} // namespace Vulkan
