// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/lru_cache.h"
#include "video_core/amdgpu/regs_texture.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <optional>
#include <utility>

namespace Vulkan {
class Instance;
class ResourceHeaps;
} // namespace Vulkan

namespace VideoCore {

/// A sampler's place in the descriptor heap's sampler heap, given back when the sampler goes.
class SamplerHeapSlot {
public:
    SamplerHeapSlot() = default;
    SamplerHeapSlot(Vulkan::ResourceHeaps* owner_, u32 index_, u32 border_)
        : owner{owner_}, index{index_}, border{border_} {}
    ~SamplerHeapSlot();

    SamplerHeapSlot(const SamplerHeapSlot&) = delete;
    SamplerHeapSlot& operator=(const SamplerHeapSlot&) = delete;

    SamplerHeapSlot(SamplerHeapSlot&& other) noexcept
        : owner{std::exchange(other.owner, nullptr)}, index{other.index}, border{other.border} {}
    SamplerHeapSlot& operator=(SamplerHeapSlot&& other) noexcept {
        if (this != &other) {
            Release();
            owner = std::exchange(other.owner, nullptr);
            index = other.index;
            border = other.border;
        }
        return *this;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return owner != nullptr;
    }
    [[nodiscard]] u32 Index() const noexcept {
        return index;
    }

private:
    void Release();

    Vulkan::ResourceHeaps* owner{};
    u32 index{};
    /// Registered custom border color, or ~0 for none.
    u32 border{};
};

class Sampler : public Common::LRUNode<> {
public:
    explicit Sampler(const Vulkan::Instance& instance, const AmdGpu::Sampler& sampler,
                     const AmdGpu::BorderColorBuffer border_color_base, bool is_depth);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    vk::Sampler Handle() const noexcept {
        return *handle;
    }

    /// The create info, chained to `custom` when it has a custom border color.
    const vk::SamplerCreateInfo& HeapCreateInfo(
        vk::SamplerCreateInfo& ci, vk::SamplerCustomBorderColorCreateInfoEXT& custom) const;

    [[nodiscard]] const std::optional<vk::SamplerCustomBorderColorCreateInfoEXT>& CustomBorder()
        const noexcept {
        return custom_border;
    }

    u64 hash;
    SamplerHeapSlot heap_slot;

private:
    vk::UniqueSampler handle;
    /// The create info without its chain, which is rebuilt by HeapCreateInfo.
    vk::SamplerCreateInfo create_info{};
    std::optional<vk::SamplerCustomBorderColorCreateInfoEXT> custom_border;
};

} // namespace VideoCore
