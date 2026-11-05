// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <deque>
#include <memory>
#include <span>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_buffer_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Shader {
struct Info;
}

namespace VideoCore {
struct ImageView;
class Sampler;
struct StreamBuffer;
struct Buffer;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;

/// How big one descriptor slot is in each heap.
struct HeapGeometry {
    u32 resource_stride{};
    u32 sampler_stride{};

    static HeapGeometry Of(const Instance& instance);
};

/// The mappings that tell the driver where one shader stage finds its descriptors: each binding
/// reads its heap index from the stage's table, whose address is in push data.
struct StageHeapMappings {
    std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings;
    vk::ShaderDescriptorSetAndBindingMappingInfoEXT info{};

    /// Fills the mappings for `stage`, its stage table included, and points `info` at them, so
    /// this must not move after.
    void Build(const Shader::Info& stage, const HeapGeometry& geometry);
};

/// The descriptors one stage of a draw writes, as the rasterizer gathered them.
struct HeapStageWrites {
    /// Logical stage, which picks the stage's table address in push data.
    u32 stage{};
    u32 first_write{};
    u32 num_writes{};
    /// One more than the highest binding the stage uses.
    u32 num_bindings{};
    std::array<u8, Shader::StageTable::BufOffsetsSize> buf_offsets{};
    std::array<u16, Shader::StageTable::StridesSize / sizeof(u16)> strides{};
    /// Formats of the buffers whose loads read them at run time.
    std::array<Shader::RuntimeBufferFormat, Shader::NUM_BUFFERS> formats{};
    /// One more than the highest buffer with a format.
    u32 num_formats{};
    /// Swizzles of storage images whose shaders read them at run time.
    std::array<u32, Shader::NUM_IMAGES> image_swizzles{};
};

/// What the rasterizer gathered for one draw or dispatch.
struct HeapDraw {
    std::span<const vk::WriteDescriptorSet> writes;
    std::span<const vk::DescriptorBufferInfo> buffer_infos;
    /// Device address of each buffer info's range, 0 for a null descriptor.
    std::span<const vk::DeviceAddress> buffer_addresses;
    std::span<const vk::DescriptorImageInfo> image_infos;
    /// The view behind each image info, null for a null descriptor or a sampler.
    std::span<const VideoCore::ImageView* const> image_views;
    /// Sampler heap slot of each image info that holds a sampler.
    std::span<const u32> sampler_slots;
    std::span<const HeapStageWrites> stages;
};

/// Shader resources through VK_EXT_descriptor_heap: a resource heap that each draw writes its
/// descriptors into as a ring, and a sampler heap with one slot per live sampler.
class ResourceHeaps {
public:
    explicit ResourceHeaps(const Instance& instance, Scheduler& scheduler,
                           VideoCore::StreamBuffer& table_buffer);
    ~ResourceHeaps();

    ResourceHeaps(const ResourceHeaps&) = delete;
    ResourceHeaps& operator=(const ResourceHeaps&) = delete;

    [[nodiscard]] const HeapGeometry& Geometry() const noexcept {
        return geometry;
    }

    /// Writes the draw's descriptors and stage tables, binds both heaps and pushes the push
    /// data. Heaps are bound again for every draw, since helper passes push descriptors.
    void Bind(const HeapDraw& draw, const Shader::PushData& push_data);

    /// The sampler's slot in the sampler heap, writing its descriptor the first time.
    u32 SamplerSlot(VideoCore::Sampler& sampler);

    /// Gives a sampler's slot back, from the sampler's destructor.
    void ReleaseSampler(u32 index, u32 border);

private:
    /// Reserves `count` resource slots for the GPU's next use and returns the first.
    u32 AllocateResources(u32 count);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::StreamBuffer& table_buffer;
    HeapGeometry geometry;

    std::unique_ptr<VideoCore::Buffer> resource_heap;
    vk::BindHeapInfoEXT resource_bind{};
    u32 resource_slots{};
    u32 resource_cursor{};
    /// Slots in use by submissions the GPU may not have finished, oldest first.
    struct Segment {
        u64 tick;
        u32 begin;
        u32 end;
    };
    std::deque<Segment> resource_segments;

    std::unique_ptr<VideoCore::Buffer> sampler_heap;
    vk::BindHeapInfoEXT sampler_bind{};
    std::vector<u32> free_samplers;

    // Scratch reused by every Bind.
    std::vector<vk::ResourceDescriptorInfoEXT> resource_infos;
    std::vector<vk::HostAddressRangeEXT> resource_ranges;
    std::vector<vk::DeviceAddressRangeEXT> address_ranges;
    std::vector<vk::ImageDescriptorInfoEXT> image_descs;
    struct ViewInfo {
        vk::ImageViewCreateInfo ci;
        vk::ImageViewUsageCreateInfo usage;
        vk::ImageViewMinLodCreateInfoEXT min_lod;
    };
    std::vector<ViewInfo> view_infos;
};

} // namespace Vulkan
