// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include <tsl/robin_map.h>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;

/// Shaders compiled once through VK_EXT_shader_object, so that every draw that uses a shader
/// shares it instead of compiling it again inside each pipeline. Modules are registered on the
/// GPU thread; shader objects can be created from any thread, and two threads that want the
/// same one wait for a single creation.
class ShaderObjects {
public:
    explicit ShaderObjects(const Instance& instance);
    ~ShaderObjects();

    ShaderObjects(const ShaderObjects&) = delete;
    ShaderObjects& operator=(const ShaderObjects&) = delete;

    /// Keeps the SPIR-V behind a module, which the shader object is made from later.
    void Register(vk::ShaderModule module, std::span<const u32> code);

    /// Drops the module's shader object and code, for a module that's being replaced. Nothing
    /// may be creating shader objects meanwhile.
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
        return {created.exchange(0, std::memory_order_relaxed),
                reused.exchange(0, std::memory_order_relaxed)};
    }

private:
    struct Slot {
        std::vector<u32> code;
        std::once_flag once;
        vk::ShaderEXT shader;
    };

    /// Creates the slot's shader object once; later callers wait for it and reuse it.
    vk::ShaderEXT Resolve(Slot& slot, vk::ShaderStageFlagBits stage, const void* next,
                          u64& driver_ns);

    vk::ShaderEXT Create(std::span<const u32> code, vk::ShaderStageFlagBits stage, const void* next,
                         u64& driver_ns);

    const Instance& instance;
    std::mutex mutex;
    tsl::robin_map<VkShaderModule, std::shared_ptr<Slot>> by_module;
    tsl::robin_map<u64, std::shared_ptr<Slot>> generated;
    std::atomic<u32> created{};
    std::atomic<u32> reused{};
};

} // namespace Vulkan
