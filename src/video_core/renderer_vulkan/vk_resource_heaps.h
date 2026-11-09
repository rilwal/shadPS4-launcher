// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <deque>
#include <memory>
#include <mutex>
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
    /// Render target swizzles, for fragment shaders that read them at run time.
    std::array<u32, 8> target_swizzles{};
    /// Vertex attribute formats, for vertex shaders that read them at run time.
    std::array<Shader::RuntimeVertexFormat, Shader::StageTable::NumVertexFormats> vertex_formats{};
    u32 num_vertex_formats{};
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
/// buffer descriptors into as a ring, and a sampler heap with one slot per live sampler. Image
/// views and null descriptors keep slots of their own at the end of the resource heap, written
/// once (SHADPS4_KEEP_DESCRIPTORS=0 writes them into the ring for every draw instead).
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

    /// Gives a kept image view slot back, from the view's destructor.
    void ReleaseKeptSlot(u32 index);

private:
    static constexpr u32 NoSlot = ~0u;

    /// The kept slot for one descriptor of a draw (an image view or a null descriptor), writing
    /// it the first time, or NoSlot when it goes in the ring.
    u32 KeptSlot(const HeapDraw& draw, const vk::WriteDescriptorSet& write, u32 element);

    /// A free kept slot, or NoSlot when they're all taken.
    u32 TakeKeptSlot();

    /// Writes one descriptor into a slot.
    void WriteSlot(u32 slot, const vk::ResourceDescriptorInfoEXT& info);

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

    /// Slots after the ring, kept by image views and null descriptors.
    bool keep_descriptors{};
    /// Views can be destroyed by deferred operations on whichever thread submits.
    std::mutex kept_mutex;
    std::vector<u32> free_kept;
    /// Null storage buffer, sampled image and storage image descriptors.
    std::array<u32, 3> null_slots{NoSlot, NoSlot, NoSlot};
    /// Each of the stage's resource descriptors' kept slot, or NoSlot for the ring.
    std::vector<u32> kept_slots;

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
