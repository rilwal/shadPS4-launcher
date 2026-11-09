// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <utility>

#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
}

namespace Shader {
struct ImageResource;
}

namespace Vulkan {
class Instance;
class Scheduler;
class ResourceHeaps;
} // namespace Vulkan

namespace VideoCore {

/// The descriptors a view keeps in the descriptor heap, one for each descriptor type and layout
/// it's bound with, so draws point at them instead of writing new ones. Given back when the view
/// goes, which is after the GPU is done with it.
class ImageViewHeapSlots {
public:
    static constexpr u32 MaxSlots = 4;

    ImageViewHeapSlots() = default;
    ~ImageViewHeapSlots();

    ImageViewHeapSlots(const ImageViewHeapSlots&) = delete;
    ImageViewHeapSlots& operator=(const ImageViewHeapSlots&) = delete;

    ImageViewHeapSlots(ImageViewHeapSlots&& other) noexcept
        : owner{std::exchange(other.owner, nullptr)}, slots{other.slots},
          count{std::exchange(other.count, 0)} {}
    ImageViewHeapSlots& operator=(ImageViewHeapSlots&& other) noexcept {
        if (this != &other) {
            Release();
            owner = std::exchange(other.owner, nullptr);
            slots = other.slots;
            count = std::exchange(other.count, 0);
        }
        return *this;
    }

    /// The slot holding this view's descriptor of that type and layout, or ~0.
    [[nodiscard]] u32 Find(vk::DescriptorType type, vk::ImageLayout layout) const noexcept {
        for (u32 i = 0; i < count; ++i) {
            if (slots[i].type == type && slots[i].layout == layout) {
                return slots[i].index;
            }
        }
        return ~0u;
    }

    [[nodiscard]] bool Full() const noexcept {
        return count == MaxSlots;
    }

    void Add(Vulkan::ResourceHeaps* owner_, u32 index, vk::DescriptorType type,
             vk::ImageLayout layout) noexcept {
        owner = owner_;
        slots[count++] = {index, type, layout};
    }

private:
    void Release();

    struct Slot {
        u32 index;
        vk::DescriptorType type;
        vk::ImageLayout layout;
    };
    Vulkan::ResourceHeaps* owner{};
    std::array<Slot, MaxSlots> slots{};
    u32 count{};
};

struct ImageViewInfo {
    ImageViewInfo() = default;
    ImageViewInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept;
    ImageViewInfo(const AmdGpu::ColorBuffer& col_buffer) noexcept;
    ImageViewInfo(const AmdGpu::DepthBuffer& depth_buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl);

    AmdGpu::ImageType type = AmdGpu::ImageType::Color2D;
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    SubresourceRange range;
    AmdGpu::CompMapping mapping{AmdGpu::IdentityMapping};
    u32 min_lod = 0;
    bool is_storage = false;

    bool operator==(const ImageViewInfo&) const = default;
};

struct Image;

struct ImageView {
    ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info, const Image& image);
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    /// What a descriptor heap needs to write a descriptor for this view: its create info,
    /// chained through the given structs that live as long as the result is used.
    const vk::ImageViewCreateInfo& HeapCreateInfo(vk::ImageViewCreateInfo& ci,
                                                  vk::ImageViewUsageCreateInfo& usage,
                                                  vk::ImageViewMinLodCreateInfoEXT& min_lod) const;

    ImageViewInfo info;
    vk::UniqueImageView image_view;
    /// Written by the descriptor heap's Bind, which only sees the view as const.
    mutable ImageViewHeapSlots heap_slots;

private:
    /// The create info without its chain, which is rebuilt by HeapCreateInfo.
    vk::ImageViewCreateInfo create_info{};
    vk::ImageUsageFlags usage_flags{};
    float min_lod{};
};

} // namespace VideoCore
